import socket

import pytest
import requests
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.research import (
    InvalidResultURLError,
    SearchBlockedError,
    SearchRateLimitError,
    SearchResponseError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
    WebPageBlockedError,
    WebPageExtractionError,
    WebPageTooLargeError,
    WebSearchService,
)


class FakeDDGS:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.options = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def text(self, query, **options):
        self.options = {"query": query, **options}
        if self.error is not None:
            raise self.error
        return self.response


def make_service(response=None, error=None, **options):
    clients = []

    def factory(**factory_options):
        client = FakeDDGS(response=response, error=error)
        client.factory_options = factory_options
        clients.append(client)
        return client

    service = WebSearchService(ddgs_factory=factory, **options)
    return service, clients


def test_search_returns_structured_results_and_passes_ddgs_options():
    service, clients = make_service(
        [
            {
                "title": "Java inheritance",
                "href": "HTTPS://Example.org/guide#section",
                "body": "A guide to inheritance.",
                "source": "Example",
                "date": "2026-01-01",
            }
        ],
        max_results=7,
        timeout_seconds=4.2,
    )

    results = service.search(
        " Java inheritance ",
        backend="duckduckgo,brave",
        region="in-en",
        safesearch="on",
        timelimit="y",
    )

    assert results == [
        SearchResult(
            title="Java inheritance",
            url="https://example.org/guide",
            snippet="A guide to inheritance.",
            source="Example",
            metadata={"date": "2026-01-01"},
        )
    ]
    assert clients[0].factory_options == {"timeout": 5}
    assert clients[0].options == {
        "query": "Java inheritance",
        "max_results": 7,
        "backend": "duckduckgo,brave",
        "region": "in-en",
        "safesearch": "on",
        "timelimit": "y",
    }


def test_search_allows_per_call_result_limit_within_safe_cap():
    service, clients = make_service(
        [{"title": f"Result {index}", "href": f"https://example.org/{index}"}
         for index in range(1, 5)]
    )

    results = service.search("topic", max_results=2)

    assert len(results) == 2
    assert clients[0].options["max_results"] == 2


def test_search_returns_empty_list_for_no_results():
    service, _ = make_service([])

    assert service.search("no matching topic") == []


def test_search_rejects_non_list_response():
    service, _ = make_service({"title": "Unexpected"})

    with pytest.raises(SearchResponseError, match="unexpected response"):
        service.search("topic")


def test_search_skips_malformed_items_but_rejects_all_malformed_results():
    service, _ = make_service(
        [
            None,
            {"title": "", "href": "https://example.org"},
            {"title": "Invalid scheme", "href": "javascript:alert(1)"},
            {"title": "Valid", "href": "https://example.org", "body": "Text"},
        ]
    )

    assert service.search("topic") == [
        SearchResult("Valid", "https://example.org/", "Text")
    ]

    service, _ = make_service([{"title": "Missing URL"}])
    with pytest.raises(SearchResponseError, match="no valid"):
        service.search("topic")


def test_search_removes_duplicate_urls_without_discarding_distinct_paths():
    service, _ = make_service(
        [
            {"title": "First", "href": "https://example.org/page#one"},
            {"title": "Duplicate", "href": "https://EXAMPLE.org/page#two"},
            {"title": "Different case-sensitive path", "href": "https://example.org/Page"},
        ]
    )

    results = service.search("topic")

    assert [result.title for result in results] == [
        "First",
        "Different case-sensitive path",
    ]


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (TimeoutException("timed out"), SearchTimeoutError),
        (TimeoutError("timed out"), SearchTimeoutError),
        (RatelimitException("slow down"), SearchRateLimitError),
        (DDGSException("captcha required"), SearchBlockedError),
        (DDGSException("connection failed"), SearchUnavailableError),
        (requests.ConnectionError("offline"), SearchUnavailableError),
    ],
)
def test_search_maps_provider_errors_to_safe_messages(error, expected_error):
    service, clients = make_service(error=error)

    with pytest.raises(expected_error) as raised:
        service.search("topic")

    assert str(raised.value)
    assert "captcha required" not in str(raised.value)
    assert len(clients) == 1


def test_search_validates_query_and_provider_options():
    service, _ = make_service([])

    with pytest.raises(ValueError, match="search query"):
        service.search("   ")
    with pytest.raises(ValueError, match="500 characters"):
        service.search("x" * 501)
    with pytest.raises(ValueError, match="between 1 and"):
        service.search("topic", max_results=21)
    with pytest.raises(ValueError, match="backend"):
        service.search("topic", backend="not-a-backend")
    with pytest.raises(ValueError, match="Safe search"):
        service.search("topic", safesearch=[])


class FakeResponse:
    def __init__(
        self,
        body,
        *,
        status_code=200,
        headers=None,
        encoding="utf-8",
    ):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = encoding
        self.is_redirect = 300 <= status_code < 400
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield self.body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, url, **options):
        self.requests.append((url, options))
        return self.response


def make_extraction_service(response, **options):
    session = FakeSession(response)
    service = WebSearchService(
        session_factory=lambda: session,
        **options,
    )
    return service, session


def allow_public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type: [
            (socket.AF_INET, type, 6, "", ("93.184.216.34", port))
        ],
    )


def test_extract_page_fetches_public_html_and_returns_readable_text(monkeypatch):
    allow_public_dns(monkeypatch)
    html = (
        "<html><head><title>Inheritance guide</title>"
        '<meta name="author" content="Faculty"></head>'
        "<body><main><h1>Inheritance</h1>"
        "<p>Inheritance allows a class to reuse behavior from another class.</p>"
        "</main></body></html>"
    ).encode()
    response = FakeResponse(html)
    service, session = make_extraction_service(response)

    page = service.extract_page("https://example.org/guide")

    assert page.url == "https://example.org/guide"
    assert page.title == "Inheritance"
    assert "Inheritance allows a class to reuse behavior" in page.text
    assert isinstance(page.metadata, dict)
    assert session.requests[0][1]["allow_redirects"] is False
    assert session.requests[0][1]["timeout"] == 8.0
    assert response.closed


def test_extract_page_rejects_private_destinations_before_http_request():
    response = FakeResponse(b"<html><body>Private</body></html>")
    service, session = make_extraction_service(response)

    with pytest.raises(InvalidResultURLError, match="public webpages"):
        service.extract_page("http://127.0.0.1/admin")

    assert session.requests == []


def test_extract_page_rejects_private_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type: [
            (socket.AF_INET, type, 6, "", ("10.0.0.8", port))
        ],
    )
    private_dns_service, private_dns_session = make_extraction_service(
        FakeResponse(b"<html><body>Private</body></html>")
    )

    with pytest.raises(InvalidResultURLError, match="public webpages"):
        private_dns_service._validate_public_destination("https://example.org")
    assert private_dns_session.requests == []


def test_extract_page_does_not_follow_redirects_to_private_addresses(
    monkeypatch,
):
    allow_public_dns(monkeypatch)
    response = FakeResponse(
        b"",
        status_code=302,
        headers={
            "Content-Type": "text/html",
            "Location": "http://127.0.0.1/admin",
        },
    )
    service, session = make_extraction_service(response)

    with pytest.raises(InvalidResultURLError, match="public webpages"):
        service.extract_page("https://example.org")

    assert len(session.requests) == 1


def test_extract_page_enforces_download_limit(monkeypatch):
    allow_public_dns(monkeypatch)
    response = FakeResponse(
        b"x" * 20,
        headers={"Content-Type": "text/html", "Content-Length": "20"},
    )
    service, _ = make_extraction_service(response, max_page_bytes=10)

    with pytest.raises(WebPageTooLargeError):
        service.extract_page("https://example.org")


def test_extract_page_maps_blocked_and_unreadable_responses(monkeypatch):
    allow_public_dns(monkeypatch)
    blocked, _ = make_extraction_service(
        FakeResponse(b"", status_code=403)
    )
    with pytest.raises(WebPageBlockedError, match="blocked"):
        blocked.extract_page("https://example.org")

    non_html, _ = make_extraction_service(
        FakeResponse(b"data", headers={"Content-Type": "application/pdf"})
    )
    with pytest.raises(WebPageExtractionError, match="not an HTML"):
        non_html.extract_page("https://example.org")
