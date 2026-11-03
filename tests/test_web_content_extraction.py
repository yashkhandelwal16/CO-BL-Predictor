from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import requests
from requests import PreparedRequest

from src.research import (
    SearchResult,
    WebContentExtractionService,
    WebContentExtractionWorker,
)
from src.research.web_content_extraction import (
    _PinnedAddressAdapter,
)


class FakeResponse:
    def __init__(self, body=b"", *, status_code=200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
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
    def __init__(self, responses=None):
        self.responses = defaultdict(deque)
        for url, response in (responses or {}).items():
            self.responses[url].extend(
                response if isinstance(response, list) else [response]
            )
        self.requests = []
        self.adapters = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def mount(self, prefix, adapter):
        self.adapters[prefix] = adapter

    def get(self, url, **options):
        self.requests.append((url, options))
        if self.responses[url]:
            return self.responses[url].popleft()
        if url.endswith("/robots.txt"):
            return FakeResponse(b"", status_code=404)
        return FakeResponse(b"<html><body>page</body></html>")


def allow_public_dns(monkeypatch):
    import src.research.url_security as url_security

    monkeypatch.setattr(
        url_security.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (url_security.socket.AF_INET, type, 6, "", ("93.184.216.34", port))
        ],
    )


def make_service(monkeypatch, responses=None, **options):
    allow_public_dns(monkeypatch)
    session = FakeSession(responses)
    service = WebContentExtractionService(
        session_factory=lambda: session,
        page_extractor=lambda _html, **_kwargs: "Readable page content.",
        metadata_extractor=lambda *_args, **_kwargs: SimpleNamespace(
            title="Page title",
            date="2025-03-02",
            sitename="Example",
            author="Faculty",
        ),
        **options,
    )
    return service, session


def result(url="https://example.org/guide", snippet="Search snippet"):
    return SearchResult("Search title", url, snippet, source="Search source")


def test_extracts_content_and_preserves_search_metadata(monkeypatch):
    service, session = make_service(monkeypatch)

    extracted = service.extract_results([result()])[0]

    assert extracted.title == "Page title"
    assert extracted.url == "https://example.org/guide"
    assert extracted.content == "Readable page content."
    assert extracted.snippet == "Search snippet"
    assert extracted.source == "Example"
    assert extracted.publication_date == "2025-03-02"
    assert extracted.metadata["author"] == "Faculty"
    assert extracted.extraction_status == "success"
    assert extracted.retrieved_at
    assert session.requests[0][0] == "https://example.org/robots.txt"
    assert session.requests[1][0] == "https://example.org/guide"
    assert session.requests[1][1]["allow_redirects"] is False
    assert session.requests[1][1]["timeout"] == 8.0
    assert "https://example.org/" in session.adapters


def test_empty_extraction_returns_original_search_fallback(monkeypatch):
    allow_public_dns(monkeypatch)
    session = FakeSession()
    service = WebContentExtractionService(
        session_factory=lambda: session,
        page_extractor=lambda *_args, **_kwargs: "  ",
        metadata_extractor=lambda *_args, **_kwargs: SimpleNamespace(),
    )

    extracted = service.extract_results([result()])[0]

    assert extracted.extraction_status == "failed"
    assert extracted.title == "Search title"
    assert extracted.url == "https://example.org/guide"
    assert extracted.content == extracted.snippet == "Search snippet"
    assert "WebPageExtractionError" in extracted.error_details


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "http://127.0.0.1/admin", "http://localhost/"]
)
def test_invalid_and_private_urls_fail_without_requests(monkeypatch, url):
    service, session = make_service(monkeypatch)

    extracted = service.extract_results([result(url)])[0]

    assert extracted.extraction_status == "failed"
    assert extracted.url == url
    assert extracted.content == "Search snippet"
    assert session.requests == []


def test_private_dns_resolution_is_rejected_before_fetch(monkeypatch):
    import src.research.url_security as url_security

    monkeypatch.setattr(
        url_security.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (url_security.socket.AF_INET, type, 6, "", ("10.0.0.8", port))
        ],
    )
    session = FakeSession()
    service = WebContentExtractionService(session_factory=lambda: session)

    extracted = service.extract_results([result()])[0]

    assert extracted.extraction_status == "failed"
    assert "public webpages" in extracted.error_details
    assert session.requests == []


def test_private_dns_and_redirect_destinations_are_rejected(monkeypatch):
    allow_public_dns(monkeypatch)
    session = FakeSession(
        {
            "https://example.org/guide": FakeResponse(
                status_code=302,
                headers={
                    "Content-Type": "text/html",
                    "Location": "http://127.0.0.1/admin",
                },
            )
        }
    )
    service = WebContentExtractionService(
        session_factory=lambda: session,
        page_extractor=lambda *_args, **_kwargs: "text",
        metadata_extractor=lambda *_args, **_kwargs: SimpleNamespace(),
    )

    extracted = service.extract_results([result()])[0]

    assert extracted.extraction_status == "failed"
    assert "public webpages" in extracted.error_details
    assert all("127.0.0.1" not in request[0] for request in session.requests)


def test_robots_policy_blocks_disallowed_pages(monkeypatch):
    robots = FakeResponse(
        b"User-agent: CO-BL-Predictor\nDisallow: /",
        headers={"Content-Type": "text/plain"},
    )
    service, session = make_service(
        monkeypatch, {"https://example.org/robots.txt": robots}
    )

    extracted = service.extract_results([result()])[0]

    assert extracted.extraction_status == "failed"
    assert "robots policy" in extracted.error_details
    assert len(session.requests) == 1


def test_timeout_and_blocked_sites_preserve_search_result(monkeypatch):
    allow_public_dns(monkeypatch)
    timeout_session = FakeSession()

    def timeout_get(url, **options):
        timeout_session.requests.append((url, options))
        if url.endswith("/robots.txt"):
            return FakeResponse(b"", status_code=404)
        raise requests.Timeout("private diagnostic")

    timeout_session.get = timeout_get
    timeout_service = WebContentExtractionService(
        session_factory=lambda: timeout_session,
        page_extractor=lambda *_args, **_kwargs: "unused",
    )
    timed_out = timeout_service.extract_results([result()])[0]
    assert timed_out.extraction_status == "failed"
    assert timed_out.content == "Search snippet"
    assert "private diagnostic" not in timed_out.error_details

    blocked_service, _ = make_service(
        monkeypatch,
        {
            "https://example.org/robots.txt": FakeResponse(
                b"", status_code=403, headers={"Content-Type": "text/plain"}
            )
        },
    )
    blocked = blocked_service.extract_results([result()])[0]
    assert blocked.extraction_status == "failed"
    assert "blocked" in blocked.error_details


def test_size_limit_and_non_html_pages_fall_back(monkeypatch):
    service, _ = make_service(
        monkeypatch,
        {
            "https://example.org/guide": FakeResponse(
                b"123456",
                headers={
                    "Content-Type": "text/html",
                    "Content-Length": "6",
                },
            )
        },
        max_response_bytes=5,
    )
    too_large = service.extract_results([result()])[0]
    assert too_large.extraction_status == "failed"
    assert "WebPageTooLargeError" in too_large.error_details

    non_html_service, _ = make_service(
        monkeypatch,
        {
            "https://example.org/guide": FakeResponse(
                b"pdf", headers={"Content-Type": "application/pdf"}
            )
        },
    )
    non_html = non_html_service.extract_results([result()])[0]
    assert non_html.extraction_status == "failed"
    assert "WebPageExtractionError" in non_html.error_details


def test_duplicate_success_uses_cache_and_retains_each_search_fallback(monkeypatch):
    service, session = make_service(monkeypatch)
    original = result("https://example.org/guide#first", "First snippet")
    duplicate = result("https://EXAMPLE.org/guide#second", "Second snippet")

    first, second = service.extract_results([original, duplicate])

    assert first.extraction_status == "success"
    assert second.extraction_status == "cached"
    assert second.snippet == "Second snippet"
    assert second.content == first.content
    assert second.retrieved_at == first.retrieved_at
    assert len(session.requests) == 2
    cached = service.extract_results([duplicate])[0]
    assert cached.extraction_status == "cached"
    assert cached.retrieved_at == first.retrieved_at


def test_failed_duplicate_keeps_its_own_fallback_and_status(monkeypatch):
    allow_public_dns(monkeypatch)
    session = FakeSession()
    original_get = session.get

    def timeout_page(url, **options):
        if url.endswith("/guide"):
            session.requests.append((url, options))
            raise requests.Timeout("timeout")
        return original_get(url, **options)

    session.get = timeout_page
    service = WebContentExtractionService(session_factory=lambda: session)
    first = result(snippet="first")
    second = result(snippet="second")

    failed_first, failed_second = service.extract_results([first, second])

    assert failed_first.extraction_status == failed_second.extraction_status == "failed"
    assert failed_first.content == "first"
    assert failed_second.content == "second"
    assert len(session.requests) == 2


def test_page_cap_skips_only_unique_results_after_limit(monkeypatch):
    service, session = make_service(monkeypatch, max_pages=1)

    extracted = service.extract_results(
        [result("https://example.org/one"), result("https://example.org/two")]
    )

    assert [item.extraction_status for item in extracted] == ["success", "skipped"]
    assert extracted[1].content == "Search snippet"
    assert len(session.requests) == 2


def test_dns_pinned_adapter_keeps_tls_hostname():
    request = PreparedRequest()
    request.prepare_url("https://example.org/guide", None)
    adapter = _PinnedAddressAdapter("93.184.216.34", "example.org")
    host_params, pool_kwargs = adapter.build_connection_pool_key_attributes(
        request, verify=True, cert=None
    )
    adapter.close()

    assert host_params["host"] == "93.184.216.34"
    assert pool_kwargs["server_hostname"] == "example.org"
    assert pool_kwargs["assert_hostname"] == "example.org"


def test_worker_returns_cancellation_results_without_fetching(monkeypatch):
    service, session = make_service(monkeypatch)
    worker = WebContentExtractionWorker(service, [result()])
    emitted = []
    worker.completed.connect(emitted.append)
    worker.cancel()

    worker.run()

    assert emitted[0][0].extraction_status == "cancelled"
    assert session.requests == []
