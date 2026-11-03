import ipaddress
import math
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
    retrieved_at: str = ""


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
class _DownloadResponse:
    body: bytes | None
    url: str
    redirect_url: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    page: ExtractedPage
    expires_at: float


class _PinnedAddressAdapter(HTTPAdapter):
    """Pin the connection IP while preserving the original TLS hostname."""

    def __init__(self, address: str, hostname: str, **kwargs: Any) -> None:
        self._address = address
        self._hostname = hostname
        super().__init__(**kwargs)

    def matches(self, address: str, hostname: str) -> bool:
        return self._address == address and self._hostname == hostname

    def build_connection_pool_key_attributes(self, request, verify, cert):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        host_params["host"] = self._address
        if host_params["scheme"] == "https":
            pool_kwargs["server_hostname"] = self._hostname
            pool_kwargs["assert_hostname"] = self._hostname
        return host_params, pool_kwargs


SessionFactory = Callable[[], requests.Session]
Clock = Callable[[], float]
PageExtractor = Callable[..., str | None]
MetadataExtractor = Callable[..., object]


class WebContentExtractionService:
    """Safely extract a bounded set of public search results."""

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
        """Extract pages while preserving the original search result on failure."""
        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise ValueError("Search results must be provided as a sequence.")

        output: list[WebContentResult] = []
        seen: dict[str, WebContentResult] = {}
        attempted_pages = 0
        for result in results:
            if not isinstance(result, SearchResult):
                raise ValueError("Each webpage must come from a validated search result.")
            if cancellation_event is not None and cancellation_event.is_set():
                output.append(self._fallback(result, "cancelled"))
                continue

            try:
                normalized_url = normalize_http_url(result.url)
            except Exception as error:
                output.append(self._fallback(result, "failed", error))
                continue

            if normalized_url in seen:
                output.append(self._duplicate_result(seen[normalized_url], result))
                continue

            cached = self._get_cached(normalized_url)
            if cached is not None:
                extracted = self._success(result, cached, "cached")
                seen[normalized_url] = extracted
                output.append(extracted)
                continue

            if attempted_pages >= self.max_pages:
                skipped = self._fallback(result, "skipped")
                seen[normalized_url] = skipped
                output.append(skipped)
                continue

            attempted_pages += 1
            try:
                page = self._extract_one(normalized_url)
            except Exception as error:
                failed = self._fallback(result, "failed", error)
                seen[normalized_url] = failed
                output.append(failed)
                continue

            self._put_cached(normalized_url, page)
            extracted = self._success(result, page, "success")
            seen[normalized_url] = extracted
            output.append(extracted)

        return output

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _extract_one(self, url: str) -> ExtractedPage:
        with self._session_factory() as session:
            self._prepare_session(session)
            html, final_url = self._fetch_page(session, url)

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
        for name in ("author", "date", "description", "sitename", "categories", "tags"):
            value = getattr(metadata, name, None)
            if value:
                details[name] = value
        title = getattr(metadata, "title", None)
        source = getattr(metadata, "sitename", None)
        return ExtractedPage(
            url=final_url,
            text=content.strip(),
            title=title.strip() if isinstance(title, str) and title.strip() else None,
            publication_date=self._normalize_publication_date(
                getattr(metadata, "date", None)
            ),
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            metadata=details,
            retrieved_at=self._timestamp(),
        )

    def _fetch_page(
        self, session: requests.Session, initial_url: str
    ) -> tuple[bytes, str]:
        current_url = initial_url
        for redirect_count in range(MAX_REDIRECTS + 1):
            parser = self._robots_for_url(session, current_url)
            if not parser.can_fetch(USER_AGENT, current_url):
                raise WebPageBlockedError(
                    "The website's robots policy disallows content extraction."
                )

            response = self._download(
                session,
                current_url,
                max_bytes=self.max_response_bytes,
                require_html=True,
            )
            if response.redirect_url is not None:
                if redirect_count == MAX_REDIRECTS:
                    raise WebPageBlockedError("The webpage redirected too many times.")
                current_url = normalize_http_url(response.redirect_url)
                self._validate_destination(current_url)
                continue
            if response.body is None:
                raise WebPageUnavailableError("The webpage returned no content.")
            return response.body, response.url
        raise WebPageBlockedError("The webpage could not be reached.")

    def _robots_for_url(
        self, session: requests.Session, page_url: str
    ) -> RobotFileParser:
        parsed = urlsplit(page_url)
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        cache_key = origin.casefold()
        with self._robots_lock:
            cached = self._robots_cache.get(cache_key)
            now = self._clock()
            if cached is not None and cached[0] > now:
                return cached[1]

            robots_url = urlunsplit(
                (parsed.scheme, parsed.netloc, "/robots.txt", "", "")
            )
            response = self._download(
                session,
                robots_url,
                max_bytes=min(self.max_response_bytes, MAX_ROBOTS_BYTES),
                require_html=False,
                allow_missing=True,
                follow_redirects=True,
            )
            parser = RobotFileParser()
            parser.set_url(robots_url)
            body = response.body or b""
            parser.parse(body.decode("utf-8", errors="replace").splitlines())
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
        follow_redirects: bool = False,
    ) -> _DownloadResponse:
        current_url = initial_url
        redirect_limit = MAX_REDIRECTS if follow_redirects else 0
        for redirect_count in range(redirect_limit + 1):
            address = self._validate_destination(current_url)
            parsed = urlsplit(current_url)
            hostname = parsed.hostname
            if hostname is None:
                raise InvalidResultURLError("The result URL is invalid.")
            origin = urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
            self._mount_pinned_adapter(
                session, current_url, origin, address, hostname
            )
            try:
                response = session.get(
                    current_url,
                    timeout=self.timeout_seconds,
                    stream=True,
                    allow_redirects=False,
                    headers={"User-Agent": USER_AGENT},
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
                    destination = normalize_http_url(urljoin(current_url, location))
                    self._validate_destination(destination)
                    if not follow_redirects:
                        return _DownloadResponse(
                            body=None, url=current_url, redirect_url=destination
                        )
                    if redirect_count == redirect_limit:
                        raise WebPageBlockedError(
                            "The webpage redirected too many times."
                        )
                    current_url = destination
                    continue
                if allow_missing and response.status_code in {404, 410}:
                    return _DownloadResponse(body=b"", url=current_url)
                if response.status_code in {401, 403, 429} or (
                    allow_missing and response.status_code >= 500
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
                return _DownloadResponse(body=bytes(body), url=current_url)
        raise WebPageBlockedError("The webpage could not be reached.")

    @staticmethod
    def _prepare_session(session: requests.Session) -> None:
        if isinstance(session, requests.Session):
            session.trust_env = False
            session.proxies.clear()

    @staticmethod
    def _mount_pinned_adapter(
        session: requests.Session,
        url: str,
        origin: str,
        address: str,
        hostname: str,
    ) -> None:
        if hasattr(session, "get_adapter"):
            existing = session.get_adapter(url)
            if isinstance(existing, _PinnedAddressAdapter):
                if existing.matches(address, hostname):
                    return
                existing.close()
        session.mount(
            origin,
            _PinnedAddressAdapter(address=address, hostname=hostname),
        )

    @staticmethod
    def _validate_destination(url: str) -> str:
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
            retrieved_at=page.retrieved_at,
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
            message = str(error) or "Extraction failed."
            error_details = f"{type(error).__name__}: {message[:500]}"
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

    @classmethod
    def _duplicate_result(
        cls, previous: WebContentResult, result: SearchResult
    ) -> WebContentResult:
        if previous.extraction_status in {"success", "cached"}:
            return WebContentResult(
                title=previous.title,
                url=result.url,
                snippet=result.snippet,
                content=previous.content,
                source=previous.source or result.source,
                metadata={**result.metadata, **previous.metadata},
                publication_date=previous.publication_date,
                retrieved_at=previous.retrieved_at,
                extraction_status="cached",
            )
        return WebContentResult(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            content=result.snippet,
            source=result.source,
            metadata=dict(result.metadata),
            publication_date=None,
            retrieved_at=cls._timestamp(),
            extraction_status=previous.extraction_status,
            error_details=previous.error_details,
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
