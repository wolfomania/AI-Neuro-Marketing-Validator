"""IP and token-based access control middleware.

Enabled via environment variables:
  NM_ACCESS_TOKEN   — if set, requires Bearer token in Authorization header
  NM_ALLOWED_IPS    — comma-separated allowlist (e.g. "1.2.3.4,5.6.7.8")

When neither is set, the middleware is a no-op (open access).
"""

import ipaddress
import logging
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessGuardConfig:
    access_token: str
    allowed_ips: frozenset[str]

    @property
    def is_enabled(self) -> bool:
        return bool(self.access_token) or bool(self.allowed_ips)


def _parse_allowed_ips(raw: str) -> frozenset[str]:
    """Normalize and validate IP strings."""
    if not raw.strip():
        return frozenset()
    result: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ipaddress.ip_address(entry)
            result.add(entry)
        except ValueError:
            logger.warning("Ignoring invalid IP in NM_ALLOWED_IPS: %s", entry)
    return frozenset(result)


def build_config(access_token: str = "", allowed_ips_raw: str = "") -> AccessGuardConfig:
    return AccessGuardConfig(
        access_token=access_token.strip(),
        allowed_ips=_parse_allowed_ips(allowed_ips_raw),
    )


class AccessGuardMiddleware(BaseHTTPMiddleware):
    """Rejects requests that don't pass IP or token checks."""

    def __init__(self, app: Callable, config: AccessGuardConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._config.is_enabled:
            return await call_next(request)

        # Always allow health check
        if request.url.path == "/api/health":
            return await call_next(request)

        # --- IP check ---
        if self._config.allowed_ips:
            client_ip = _extract_client_ip(request)
            if client_ip not in self._config.allowed_ips:
                logger.warning("Blocked request from IP %s", client_ip)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Access denied: IP not allowed."},
                )

        # --- Token check ---
        if self._config.access_token:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing Bearer token."},
                )
            token = auth_header[len("Bearer "):]
            if token != self._config.access_token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid access token."},
                )

        return await call_next(request)


def _extract_client_ip(request: Request) -> str:
    """Get client IP, respecting X-Forwarded-For from trusted proxies."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    if client is not None:
        return client.host
    return ""
