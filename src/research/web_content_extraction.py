import ipaddress
import math
import re
import socket
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Event, RLock
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from trafilatura.metadata import extract_metadata

from src.research.url_security import (
    InvalidResultURLError,
    normalize_http_url,
    resolve_public_address,
)
from src.research.web_search import SearchResult


DEFAULT_MAX_PAGES = 3
MAX_PAGES_LIMIT = 10
DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_CACHE_SECONDS = 86_400
DEFAULT_CACHE_ENTRIES = 128
MAX_ROBOTS_BYTES = 256_000
MAX_REDIRECTS = 3
USER_AGENT = "CO-BL-Predictor"
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class WebContentError(RuntimeError):
    """Base class for webpage fetching and extraction errors."""


class WebPageTimeoutError(WebContentError):
    """Raised when a webpage request exceeds the configured timeout."""


class WebPageBlockedError(WebContentError):
    """Raised when a webpage or its robots policy blocks extraction."""


class WebPageTooLargeError(WebContentError):
    """Raised when a response exceeds the configured download limit."""


class WebPageExtractionError(WebContentError):
    """Raised when a page is not HTML or contains no readable content."""


class WebPageUnavailableError(WebContentError):
    """Raised when a page cannot be reached or downloaded."""


@dataclass(frozen=True)
class ExtractedPage:
    url: str
    text: str
    title: str | None = None
    publication_date: str | None = None
    source: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WebContentResult:
    title: str
    url: str
    snippet: str
    content: str
    source: str | None
    metadata: Mapping[str, object]
    publication_date: str | None
    retrieved_at: str
    extraction_status: Literal[
        "success", "cached", "failed", "skipped", "cancelled"
    ]
    error_details: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    page: ExtractedPage
    expires_at: float


class _PinnedAddressAdapter(HTTPAdapter):
    """Connect to a previously validated IP while retaining HTTPS host checks."""

    def __init__(self, tls_hostname: str, **kwargs: Any) -> None:
        self._tls_hostname = tls_hostname
        super().__init__(**kwargs)

    def build_connection_pool_key_attributes(self, request, verify, cert):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        if urlsplit(request.url).scheme == "https":
            pool_kwargs["server_hostname"] = self._tls_hostname
            pool_kwargs["assert_hostname"] = self._tls_hostname
        return host_params, pool_kwargs


SessionFactory = Callable[[], requests.Session]
Clock = Callable[[], float]
PageExtractor = Callable[..., str | None]
MetadataExtractor = Callable[..., object]


class WebContentExtractionService:
    """Safely extract selected public search results without requiring the UI."""

    def __init__(
        self,
        *,
        max_pages: int = DEFAULT_MAX_PAGES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        cache_seconds: int = DEFAULT_CACHE_SECONDS,
        max_cache_entries: int = DEFAULT_CACHE_ENTRIES,
        session_factory: SessionFactory | None = None,
        page_extractor: PageExtractor | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.max_pages = self._validate_max_pages(max_pages)
        self.timeout_seconds = self._validate_timeout(timeout_seconds)
        self.max_response_bytes = self._validate_positive_integer(
            max_response_bytes, "The response size limit"
        )
        self.cache_seconds = self._validate_positive_integer(
            cache_seconds, "The cache duration"
        )
        self.max_cache_entries = self._validate_positive_integer(
            max_cache_entries, "The cache size"
        )
        self._session_factory = session_factory or requests.Session
        self._page_extractor = page_extractor or trafilatura.extract
        self._metadata_extractor = metadata_extractor or extract_metadata
        self._clock = clock or time.monotonic
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._robots_cache: dict[str, tuple[float, RobotFileParser]] = {}
        self._cache_lock = RLock()
        self._robots_lock = RLock()

    def extract_results(
        self,
        results: Sequence[SearchResult],
        *,
        cancellation_event: Event | None = None,
    ) -> list[WebContentResult]:
        """Extract a bounded set while retaining search-result fallbacks."""
        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise ValueError("Search results must be provided as a sequence.")

        output: list[WebContentResult] = []
        seen: dict[str, WebContentResult] = {}
        attempted_pages = 0
        for result in results:
            if not isinstance(result, SearchResult):
                raise ValueError("Each webpage must come from a validated search result.")

            try:
                normalized_url = normalize_http_url(result.url)
                self._validate_destination(normalized_url)
            except Exception as error:
                output.append(self._fallback(result, "failed", error))
                continue

            key = normalized_url
            if key in seen:
                previous = seen[key]
                output.append(
                    self._with_search_result(previous, result, "cached")
                )
                continue

            cached = self._get_cached(key)
            if cached is not None:
                extracted = self._success(result, cached, "cached")
                seen[key] = extracted
                output.append(extracted)
                continue

            if cancellation_event is not None and cancellation_event.is_set():
                output.append(self._fallback(result, "cancelled"))
                continue
            if attempted_pages >= self.max_pages:
                skipped = self._fallback(result, "skipped")
                seen[key] = skipped
                output.append(skipped)
                continue

            attempted_pages += 1
            try:
                page = self._extract_one(normalized_url)
            except Exception as error:
                failed = self._fallback(result, "failed", error)
                seen[key] = failed
                output.append(failed)
                continue

            self._put_cached(key, page)
            extracted = self._success(result, page, "success")
            seen[key] = extracted
            output.append(extracted)

        return output

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _extract_one(self, url: str) -> ExtractedPage:
        with self._session_factory() as session:
            self._prepare_session(session)
            final_url = self._check_robots_and_get_final_url(session, url)
            html, final_url = self._download(
                session,
                final_url,
                max_bytes=self.max_response_bytes,
                require_html=True,
            )

        try:
            content = self._page_extractor(
                html,
                url=final_url,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if not isinstance(content, str) or not content.strip():
                raise WebPageExtractionError(
                    "No readable article or documentation content was found."
                )
            metadata = self._metadata_extractor(html, default_url=final_url)
        except WebContentError:
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
        publication_date = self._normalize_publication_date(
            getattr(metadata, "date", None)
        )
        return ExtractedPage(
            url=final_url,
            text=content.strip(),
            title=title.strip() if isinstance(title, str) and title.strip() else None,
            publication_date=publication_date,
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            metadata=details,
        )

    def _check_robots_and_get_final_url(
        self, session: requests.Session, url: str
    ) -> str:
        current_url = url
        for redirect_count in range(MAX_REDIRECTS + 1):
            parsed = urlsplit(current_url)
            robots_url = urlunsplit(
                (parsed.scheme, parsed.netloc, "/robots.txt", "", "")
            )
            parser = self._robots_for_url(session, robots_url)
            if not parser.can_fetch(USER_AGENT, current_url):
                raise WebPageBlockedError(
                    "The website's robots policy disallows content extraction."
                )

            final_url = self._download(
                session,
                current_url,
                max_bytes=self.max_response_bytes,
                require_html=True,
                follow_redirects=False,
                return_redirect=True,
            )
            if isinstance(final_url, tuple):
                _, location = final_url
                if redirect_count == MAX_REDIRECTS:
                    raise WebPageBlockedError(
                        "The webpage redirected too many times."
                    )
                current_url = normalize_http_url(urljoin(current_url, location))
                self._validate_destination(current_url)
                continue
            return current_url
        raise WebPageBlockedError("The webpage could not be reached.")

    def _robots_for_url(
        self, session: requests.Session, robots_url: str
    ) -> RobotFileParser:
        parsed = urlsplit(robots_url)
        cache_key = f"{parsed.scheme}://{parsed.netloc}"
        with self._robots_lock:
            cached = self._robots_cache.get(cache_key)
            now = self._clock()
            if cached is not None and cached[0] > now:
                return cached[1]

            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                body, _ = self._download(
                    session,
                    robots_url,
                    max_bytes=min(self.max_response_bytes, MAX_ROBOTS_BYTES),
                    require_html=False,
                    allow_missing=True,
                )
                parser.parse(body.decode("utf-8", errors="replace").splitlines())
            except WebPageUnavailableError:
                parser.parse([])
            self._robots_cache[cache_key] = (
                now + min(self.cache_seconds, 3600),
                parser,
            )
            return parser

    def _download(
        self,
        session: requests.Session,
        initial_url: str,
        *,
        max_bytes: int,
        require_html: bool,
        allow_missing: bool = False,
        follow_redirects: bool = True,
        return_redirect: bool = False,
    ) -> tuple[bytes, str] | tuple[str, str] | bytes:
        current_url = initial_url
        redirect_limit = MAX_REDIRECTS if follow_redirects else 0
        for redirect_count in range(redirect_limit + 1):
            parsed = urlsplit(current_url)
            host = parsed.hostname
            if host is None:
                raise InvalidResultURLError("The result URL is invalid.")
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            address = self._validate_destination(current_url)
            request_url = self._url_with_address(current_url, address)
            request_headers = {"User-Agent": USER_AGENT}
            request_headers["Host"] = self._host_header(parsed)
            try:
                response = session.get(
                    request_url,
                    timeout=self.timeout_seconds,
                    stream=True,
                    allow_redirects=False,
                    headers=request_headers,
                )
            except requests.Timeout as error:
                raise WebPageTimeoutError(
                    "The webpage took too long to load."
                ) from error
            except requests.RequestException as error:
                raise WebPageUnavailableError(
                    "The webpage could not be reached or downloaded."
                ) from error
            except OSError as error:
                raise WebPageUnavailableError(
                    "The webpage could not be reached or downloaded."
                ) from error

            with response:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise WebPageBlockedError(
                            "The webpage redirected without a destination."
                        )
                    if return_redirect:
                        return current_url, location
                    if redirect_count == redirect_limit:
                        raise WebPageBlockedError(
                            "The webpage redirected too many times."
                        )
                    current_url = normalize_http_url(
                        urljoin(current_url, location)
                    )
                    self._validate_destination(current_url)
                    continue
                if response.status_code == 404 and allow_missing:
                    return b""
                if response.status_code in {401, 403, 429} or (
                    response.status_code >= 500 and allow_missing
                ):
                    raise WebPageBlockedError(
                        "The website blocked the content extraction request."
                    )
                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    raise WebPageUnavailableError(
                        "The webpage returned an unsuccessful response."
                    ) from error

                content_type = response.headers.get("Content-Type", "").casefold()
                if require_html and not any(
                    supported in content_type for supported in _HTML_CONTENT_TYPES
                ):
                    raise WebPageExtractionError(
                        "This result is not an HTML webpage."
                    )
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise WebPageTooLargeError(
                                "The webpage exceeds the allowed response size."
                            )
                    except ValueError:
                        pass
                body = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise WebPageTooLargeError(
                                "The webpage exceeds the allowed response size."
                            )
                except requests.Timeout as error:
                    raise WebPageTimeoutError(
                        "The webpage took too long to download."
                    ) from error
                except requests.RequestException as error:
                    raise WebPageUnavailableError(
                        "The webpage could not be downloaded completely."
                    ) from error
                if not require_html:
                    return bytes(body)
                return bytes(body), current_url
        raise WebPageBlockedError("The webpage could not be reached.")

    def _prepare_session(self, session: requests.Session) -> None:
        if isinstance(session, requests.Session):
            session.trust_env = False

    def _validate_destination(self, url: str) -> str:
        normalized = normalize_http_url(url)
        parsed = urlsplit(normalized)
        host = parsed.hostname
        if host is None:
            raise InvalidResultURLError("The result URL is invalid.")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            return resolve_public_address(host, port)
        except InvalidResultURLError:
            raise
        except OSError as error:
            raise WebPageUnavailableError(
                "The webpage address could not be resolved."
            ) from error

    @staticmethod
    def _url_with_address(url: str, address: str) -> str:
        parsed = urlsplit(url)
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as error:
            raise InvalidResultURLError("The result URL is invalid.") from error
        host = f"[{ip}]" if ip.version == 6 else str(ip)
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, "")
        )

    @staticmethod
    def _host_header(parsed) -> str:
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return host

    @staticmethod
    def _normalize_publication_date(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip()
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError:
                return normalized
        return parsed.date().isoformat()

    def _get_cached(self, key: str) -> ExtractedPage | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry.page

    def _put_cached(self, key: str, page: ExtractedPage) -> None:
        with self._cache_lock:
            self._cache[key] = _CacheEntry(
                page=page,
                expires_at=self._clock() + self.cache_seconds,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)

    @staticmethod
    def _success(
        result: SearchResult,
        page: ExtractedPage,
        status: Literal["success", "cached"],
    ) -> WebContentResult:
        metadata = dict(result.metadata)
        metadata.update(page.metadata)
        return WebContentResult(
            title=page.title or result.title,
            url=result.url,
            snippet=result.snippet,
            content=page.text,
            source=page.source or result.source,
            metadata=metadata,
            publication_date=page.publication_date,
            retrieved_at=WebContentExtractionService._timestamp(),
            extraction_status=status,
        )

    @staticmethod
    def _fallback(
        result: SearchResult,
        status: Literal["failed", "skipped", "cancelled"],
        error: Exception | None = None,
    ) -> WebContentResult:
        error_details = None
        if error is not None:
            error_details = (
                f"{type(error).__name__}: {str(error) or 'Extraction failed.'}"
            )
        return WebContentResult(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            content=result.snippet,
            source=result.source,
            metadata=dict(result.metadata),
            publication_date=None,
            retrieved_at=WebContentExtractionService._timestamp(),
            extraction_status=status,
            error_details=error_details,
        )

    @staticmethod
    def _with_search_result(
        cached: WebContentResult,
        result: SearchResult,
        status: Literal["cached"],
    ) -> WebContentResult:
        return WebContentResult(
            title=cached.title if cached.extraction_status != "failed" else result.title,
            url=result.url,
            snippet=result.snippet,
            content=cached.content,
            source=cached.source or result.source,
            metadata={**result.metadata, **cached.metadata},
            publication_date=cached.publication_date,
            retrieved_at=WebContentExtractionService._timestamp(),
            extraction_status=status,
            error_details=cached.error_details,
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _validate_max_pages(value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_PAGES_LIMIT
        ):
            raise ValueError(
                f"Maximum pages must be between 1 and {MAX_PAGES_LIMIT}."
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
    def _validate_positive_integer(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
        return value
