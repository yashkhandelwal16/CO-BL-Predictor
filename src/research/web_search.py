import ipaddress
import math
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from trafilatura.metadata import extract_metadata


DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_PAGE_BYTES = 2_000_000
MAX_REDIRECTS = 3
_BACKEND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:,[a-z][a-z0-9_-]*)*$")
_REGION_PATTERN = re.compile(r"^[a-z]{2}-[a-z]{2,}$")
_SUPPORTED_BACKENDS = frozenset(
    {
        "all",
        "auto",
        "brave",
        "duckduckgo",
        "grokipedia",
        "mojeek",
        "startpage",
        "wikipedia",
        "yahoo",
    }
)
_BLOCKED_SEARCH_MARKERS = (
    "captcha",
    "access denied",
    "automated queries",
    "bot detection",
    "blocked",
    "rate limit",
    "too many requests",
    "unusual traffic",
)


class DDGSClient(Protocol):
    def __enter__(self) -> "DDGSClient": ...

    def __exit__(self, *args: object) -> None: ...

    def text(
        self,
        query: str,
        *,
        max_results: int,
        backend: str,
        region: str,
        safesearch: str,
        timelimit: str | None,
    ) -> list[dict[str, Any]]: ...


DDGSFactory = Callable[..., DDGSClient]
SessionFactory = Callable[[], requests.Session]


class WebSearchError(RuntimeError):
    """Base class for safe, user-facing search and webpage errors."""


class SearchTimeoutError(WebSearchError):
    """Raised when a search request exceeds its configured timeout."""


class SearchRateLimitError(WebSearchError):
    """Raised when a search provider rate-limits a request."""


class SearchBlockedError(WebSearchError):
    """Raised when a search provider blocks a request or presents a CAPTCHA."""


class SearchResponseError(WebSearchError):
    """Raised when the search provider returns an unexpected response."""


class SearchUnavailableError(WebSearchError):
    """Raised when search cannot connect to a provider."""


class WebPageError(WebSearchError):
    """Base class for safe webpage fetching and extraction errors."""


class InvalidResultURLError(WebPageError):
    """Raised when a search result URL is invalid or points to a private host."""


class WebPageTimeoutError(WebPageError):
    """Raised when fetching a webpage exceeds its configured timeout."""


class WebPageBlockedError(WebPageError):
    """Raised when a webpage blocks the extraction request."""


class WebPageTooLargeError(WebPageError):
    """Raised when a webpage exceeds the configured download limit."""


class WebPageExtractionError(WebPageError):
    """Raised when webpage content cannot be extracted."""


class WebPageUnavailableError(WebPageError):
    """Raised when a webpage cannot be reached."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedPage:
    url: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class WebSearchService:
    """Optional DDGS search and bounded extraction of public webpages."""

    def __init__(
        self,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        backend: str = "auto",
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
        ddgs_factory: DDGSFactory | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.max_results = self._validate_max_results(max_results)
        self.timeout_seconds = self._validate_timeout(timeout_seconds)
        self.backend = self._validate_backend(backend)
        self.region = self._validate_region(region)
        self.safesearch = self._validate_safesearch(safesearch)
        self.timelimit = self._validate_timelimit(timelimit)
        if (
            isinstance(max_page_bytes, bool)
            or not isinstance(max_page_bytes, int)
            or max_page_bytes < 1
        ):
            raise ValueError("The webpage size limit must be a positive integer.")
        self.max_page_bytes = max_page_bytes
        self._ddgs_factory = ddgs_factory or DDGS
        self._session_factory = session_factory or requests.Session

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        backend: str | None = None,
        region: str | None = None,
        safesearch: str | None = None,
        timelimit: str | None = None,
    ) -> list[SearchResult]:
        """Search the public web once and return validated, unique results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Enter a search query.")
        normalized_query = query.strip()
        if len(normalized_query) > 500:
            raise ValueError("Search queries must be 500 characters or fewer.")

        result_limit = self._validate_max_results(
            self.max_results if max_results is None else max_results
        )
        selected_backend = self._validate_backend(
            self.backend if backend is None else backend
        )
        selected_region = self._validate_region(
            self.region if region is None else region
        )
        selected_safesearch = self._validate_safesearch(
            self.safesearch if safesearch is None else safesearch
        )
        selected_timelimit = self._validate_timelimit(
            self.timelimit if timelimit is None else timelimit
        )

        try:
            with self._ddgs_factory(
                timeout=max(1, math.ceil(self.timeout_seconds))
            ) as ddgs:
                raw_results = ddgs.text(
                    normalized_query,
                    max_results=result_limit,
                    backend=selected_backend,
                    region=selected_region,
                    safesearch=selected_safesearch,
                    timelimit=selected_timelimit,
                )
        except RatelimitException as error:
            raise SearchRateLimitError(
                "The search provider is temporarily rate-limiting requests. "
                "Please try again later."
            ) from error
        except TimeoutException as error:
            raise SearchTimeoutError(
                "The search took too long. Check your connection and try again."
            ) from error
        except DDGSException as error:
            if self._is_blocked_error(error):
                raise SearchBlockedError(
                    "The search provider blocked this request. Please try again later."
                ) from error
            raise SearchUnavailableError(
                "Web search is unavailable. Check your internet connection and "
                "try again later."
            ) from error
        except (requests.Timeout, TimeoutError) as error:
            raise SearchTimeoutError(
                "The search took too long. Check your connection and try again."
            ) from error
        except (requests.ConnectionError, OSError) as error:
            raise SearchUnavailableError(
                "Web search is unavailable. Check your internet connection and "
                "try again later."
            ) from error
        except Exception as error:
            raise SearchUnavailableError(
                "Web search could not be completed. Please try again later."
            ) from error

        if not isinstance(raw_results, list):
            raise SearchResponseError(
                "The search provider returned an unexpected response."
            )
        if not raw_results:
            return []

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        malformed_count = 0
        for item in raw_results:
            result = self._parse_result(item)
            if result is None:
                malformed_count += 1
                continue
            canonical_url = result.url
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)
            results.append(result)
            if len(results) == result_limit:
                break

        if not results and malformed_count:
            raise SearchResponseError(
                "The search provider returned no valid search results."
            )
        return results

    def extract_page(self, url: str) -> ExtractedPage:
        """Fetch and extract readable text from one public webpage."""
        current_url = self._normalize_url(url)
        try:
            with self._session_factory() as session:
                html, final_url = self._download_html(session, current_url)
        except InvalidResultURLError:
            raise
        except WebPageError:
            raise
        except requests.Timeout as error:
            raise WebPageTimeoutError(
                "The webpage took too long to load."
            ) from error
        except requests.Timeout as error:
            raise WebPageTimeoutError(
                "The webpage took too long to download."
            ) from error
        except requests.RequestException as error:
            raise WebPageUnavailableError(
                "The webpage could not be reached. Check your connection and "
                "try again later."
            ) from error
        except OSError as error:
            raise WebPageUnavailableError(
                "The webpage could not be reached. Check your connection and "
                "try again later."
            ) from error

        try:
            content = trafilatura.extract(
                html,
                url=final_url,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if not isinstance(content, str) or not content.strip():
                raise WebPageExtractionError(
                    "No readable article text could be extracted from this page."
                )
            metadata = extract_metadata(html, default_url=final_url)
        except WebPageError:
            raise
        except Exception as error:
            raise WebPageExtractionError(
                "Readable content could not be extracted from this webpage."
            ) from error

        details: dict[str, object] = {}
        for name in (
            "author",
            "date",
            "description",
            "sitename",
            "categories",
            "tags",
        ):
            value = getattr(metadata, name, None)
            if value:
                details[name] = value
        title = getattr(metadata, "title", None)
        source = getattr(metadata, "sitename", None)
        return ExtractedPage(
            url=final_url,
            text=content.strip(),
            title=title.strip() if isinstance(title, str) and title.strip() else None,
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            metadata=details,
        )

    def _download_html(
        self, session: requests.Session, initial_url: str
    ) -> tuple[str, str]:
        current_url = initial_url
        for redirect_count in range(MAX_REDIRECTS + 1):
            self._validate_public_destination(current_url)
            try:
                response = session.get(
                    current_url,
                    timeout=self.timeout_seconds,
                    stream=True,
                    allow_redirects=False,
                    headers={"User-Agent": "CO-BL-Predictor/1.0"},
                )
            except requests.Timeout as error:
                raise WebPageTimeoutError(
                    "The webpage took too long to load."
                ) from error
            except requests.RequestException as error:
                raise WebPageUnavailableError(
                    "The webpage could not be reached. Check your connection "
                    "and try again later."
                ) from error

            with response:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise WebPageBlockedError(
                            "The webpage redirected without a valid destination."
                        )
                    if redirect_count == MAX_REDIRECTS:
                        raise WebPageBlockedError(
                            "The webpage redirected too many times."
                        )
                    current_url = self._normalize_url(
                        urljoin(current_url, location)
                    )
                    continue
                if response.status_code == 429:
                    raise WebPageBlockedError(
                        "The webpage is temporarily rate-limiting requests."
                    )
                if response.status_code in {401, 403, 503}:
                    raise WebPageBlockedError(
                        "The webpage blocked the extraction request."
                    )
                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    raise WebPageUnavailableError(
                        "The webpage returned an unsuccessful response."
                    ) from error
                content_type = response.headers.get("Content-Type", "").casefold()
                if not any(
                    media_type in content_type
                    for media_type in ("text/html", "application/xhtml+xml")
                ):
                    raise WebPageExtractionError(
                        "This result is not an HTML webpage."
                    )

                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > self.max_page_bytes:
                            raise WebPageTooLargeError(
                                "The webpage is larger than the allowed download limit."
                            )
                    except ValueError:
                        pass
                chunks = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        chunks.extend(chunk)
                        if len(chunks) > self.max_page_bytes:
                            raise WebPageTooLargeError(
                                "The webpage is larger than the allowed download limit."
                            )
                except requests.RequestException as error:
                    raise WebPageUnavailableError(
                        "The webpage could not be downloaded completely."
                    ) from error
                encoding = response.encoding or "utf-8"
                try:
                    html = bytes(chunks).decode(encoding, errors="replace")
                except LookupError:
                    html = bytes(chunks).decode("utf-8", errors="replace")
                return html, current_url

        raise WebPageBlockedError("The webpage could not be reached.")

    @staticmethod
    def _parse_result(item: object) -> SearchResult | None:
        if not isinstance(item, Mapping):
            return None
        title = item.get("title")
        raw_url = item.get("href") or item.get("url")
        snippet = item.get("body") or item.get("snippet") or ""
        source = item.get("source")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(raw_url, str)
            or not isinstance(snippet, str)
            or (source is not None and not isinstance(source, str))
        ):
            return None
        try:
            normalized_url = WebSearchService._normalize_url(raw_url)
        except InvalidResultURLError:
            return None
        metadata = {
            str(key): value
            for key, value in item.items()
            if key not in {"title", "href", "url", "body", "snippet", "source"}
            and isinstance(key, str)
        }
        return SearchResult(
            title=title.strip(),
            url=normalized_url,
            snippet=snippet.strip(),
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_url(url: str) -> str:
        if (
            not isinstance(url, str)
            or not url
            or len(url) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in url)
        ):
            raise InvalidResultURLError("The result URL is invalid.")
        try:
            parsed = urlsplit(url.strip())
            scheme = parsed.scheme.casefold()
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as error:
            raise InvalidResultURLError("The result URL is invalid.") from error
        if (
            scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise InvalidResultURLError("The result URL is invalid.")

        normalized_host = hostname.rstrip(".").casefold()
        try:
            address = ipaddress.ip_address(normalized_host)
        except ValueError:
            try:
                normalized_host = normalized_host.encode("idna").decode("ascii")
            except UnicodeError as error:
                raise InvalidResultURLError("The result URL is invalid.") from error
        else:
            if address.version == 6:
                normalized_host = f"[{normalized_host}]"
        if not re.fullmatch(r"[a-z0-9.\-\[\]:]+", normalized_host):
            raise InvalidResultURLError("The result URL is invalid.")

        netloc = normalized_host
        if port is not None and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            netloc = f"{netloc}:{port}"
        path = parsed.path or "/"
        return urlunsplit((scheme, netloc, path, parsed.query, ""))

    @classmethod
    def _validate_public_destination(cls, url: str) -> None:
        normalized_url = cls._normalize_url(url)
        parsed = urlsplit(normalized_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if host is None or host == "localhost" or host.endswith(
            (".localhost", ".local", ".internal", ".test", ".invalid")
        ):
            raise InvalidResultURLError(
                "Only public webpages can be extracted."
            )
        try:
            literal_address = ipaddress.ip_address(host)
        except ValueError:
            try:
                addresses = socket.getaddrinfo(
                    host, port, type=socket.SOCK_STREAM
                )
            except OSError as error:
                raise WebPageUnavailableError(
                    "The webpage address could not be resolved."
                ) from error
            resolved_addresses = {
                ipaddress.ip_address(address_info[4][0].split("%", 1)[0])
                for address_info in addresses
            }
            if not resolved_addresses or any(
                not address.is_global for address in resolved_addresses
            ):
                raise InvalidResultURLError(
                    "Only public webpages can be extracted."
                )
        else:
            if not literal_address.is_global:
                raise InvalidResultURLError(
                    "Only public webpages can be extracted."
                )

    @staticmethod
    def _is_blocked_error(error: DDGSException) -> bool:
        message = str(error).casefold()
        return any(marker in message for marker in _BLOCKED_SEARCH_MARKERS)

    @staticmethod
    def _validate_max_results(value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_RESULTS_LIMIT
        ):
            raise ValueError(
                f"Maximum results must be between 1 and {MAX_RESULTS_LIMIT}."
            )
        return value

    @staticmethod
    def _validate_timeout(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"Timeout must be greater than 0 and no more than "
                f"{MAX_TIMEOUT_SECONDS:g} seconds."
            )
        return float(value)

    @staticmethod
    def _validate_backend(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("The search backend option is invalid.")
        normalized = value.casefold()
        if not _BACKEND_PATTERN.fullmatch(normalized) or any(
            backend not in _SUPPORTED_BACKENDS
            for backend in normalized.split(",")
        ):
            raise ValueError("The search backend option is invalid.")
        return normalized

    @staticmethod
    def _validate_region(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("The search region option is invalid.")
        normalized = value.casefold()
        if not _REGION_PATTERN.fullmatch(normalized):
            raise ValueError("The search region option is invalid.")
        return normalized

    @staticmethod
    def _validate_safesearch(value: str) -> str:
        if not isinstance(value, str) or value not in {"on", "moderate", "off"}:
            raise ValueError("Safe search must be on, moderate, or off.")
        return value

    @staticmethod
    def _validate_timelimit(value: str | None) -> str | None:
        if value is not None and (
            not isinstance(value, str) or value not in {"d", "w", "m", "y"}
        ):
            raise ValueError("Search time limit must be d, w, m, y, or None.")
        return value
