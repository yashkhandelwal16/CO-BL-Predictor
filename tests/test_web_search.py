import requests
import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.research import (
    SearchBlockedError,
    SearchRateLimitError,
    SearchResponseError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
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


def test_search_applies_per_call_result_limit():
    service, clients = make_service(
        [
            {"title": f"Result {index}", "href": f"https://example.org/{index}"}
            for index in range(1, 5)
        ]
    )

    results = service.search("topic", max_results=2)

    assert len(results) == 2
    assert clients[0].options["max_results"] == 2


def test_search_returns_empty_list_for_no_results():
    service, _ = make_service([])

    assert service.search("no matching topic") == []


def test_search_rejects_malformed_provider_responses():
    service, _ = make_service({"title": "Unexpected"})
    with pytest.raises(SearchResponseError, match="unexpected response"):
        service.search("topic")

    service, _ = make_service(
        [
            None,
            {"title": "", "href": "https://example.org"},
            {"title": "Invalid scheme", "href": "javascript:alert(1)"},
        ]
    )
    with pytest.raises(SearchResponseError, match="no valid"):
        service.search("topic")


def test_search_removes_duplicate_urls_without_discarding_distinct_paths():
    service, _ = make_service(
        [
            {"title": "First", "href": "https://example.org/page#one"},
            {"title": "Duplicate", "href": "https://EXAMPLE.org/page#two"},
            {
                "title": "Different case-sensitive path",
                "href": "https://example.org/Page",
            },
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
