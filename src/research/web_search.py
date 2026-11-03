import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from src.research.url_security import InvalidResultURLError, normalize_http_url


DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_TIMEOUT_SECONDS = 60.0
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


class WebSearchError(RuntimeError):
    """Base class for safe, user-facing search errors."""


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


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class WebSearchService:
    """Optional DDGS text search returning validated public result URLs."""

    def __init__(
        self,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        backend: str = "auto",
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        ddgs_factory: DDGSFactory | None = None,
    ) -> None:
        self.max_results = self._validate_max_results(max_results)
        self.timeout_seconds = self._validate_timeout(timeout_seconds)
        self.backend = self._validate_backend(backend)
        self.region = self._validate_region(region)
        self.safesearch = self._validate_safesearch(safesearch)
        self.timelimit = self._validate_timelimit(timelimit)
        self._ddgs_factory = ddgs_factory or DDGS

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
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
            if len(results) == result_limit:
                break

        if not results and malformed_count:
            raise SearchResponseError(
                "The search provider returned no valid search results."
            )
        return results

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
            normalized_url = normalize_http_url(raw_url)
        except InvalidResultURLError:
            return None
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"title", "href", "url", "body", "snippet", "source"}
            and isinstance(key, str)
        }
        return SearchResult(
            title=title.strip(),
            url=normalized_url,
            snippet=snippet.strip(),
            source=source.strip() if source and source.strip() else None,
            metadata=metadata,
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
