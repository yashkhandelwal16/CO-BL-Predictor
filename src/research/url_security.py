import ipaddress
import re
import socket
from urllib.parse import urlsplit, urlunsplit


class InvalidResultURLError(ValueError):
    """Raised when a webpage URL is invalid or targets a non-public host."""


def normalize_http_url(url: str) -> str:
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
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def resolve_public_address(hostname: str, port: int) -> str:
    host = hostname.strip("[]").rstrip(".").casefold()
    if host == "localhost" or host.endswith(
        (".localhost", ".local", ".internal", ".test", ".invalid")
    ):
        raise InvalidResultURLError("Only public webpages can be extracted.")
    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        try:
            addresses = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
        except OSError as error:
            raise OSError("The webpage address could not be resolved.") from error
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
        return str(sorted(resolved_addresses, key=lambda value: value.version)[0])
    if not literal_address.is_global:
        raise InvalidResultURLError("Only public webpages can be extracted.")
    return str(literal_address)
