"""Security headers middleware enforcing OWASP recommended HTTP response protections."""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

DEFAULT_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}

DEFAULT_HSTS_VALUE: str = "max-age=31536000; includeSubDomains"


DOCS_CSP: str = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://cdn.jsdelivr.net data:; "
    "connect-src 'self'; "
    "worker-src 'self' blob:;"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects hardened HTTP security headers into every outgoing response."""

    def __init__(
        self,
        app: ASGIApp,
        headers: dict[str, str] | None = None,
        enabled: bool = True,
        trusted_proxies: list[str] | set[str] | None = None,
        enable_hsts: bool = True,
        hsts_value: str = DEFAULT_HSTS_VALUE,
    ) -> None:
        super().__init__(app)
        self.headers = dict(headers or DEFAULT_SECURITY_HEADERS)
        if "Strict-Transport-Security" in self.headers:
            self.hsts_value = self.headers.pop("Strict-Transport-Security")
        else:
            self.hsts_value = hsts_value
        self.enabled = enabled
        self.enable_hsts = enable_hsts
        self.trusted_proxies = set(trusted_proxies or ["127.0.0.1", "::1"])

    def _is_secure(self, request: Request) -> bool:
        """Determine whether the request arrived via true HTTPS or a verified trusted proxy."""
        if request.url.scheme == "https":
            return True

        client_host = request.client.host if request.client else None
        if client_host and client_host in self.trusted_proxies:
            forwarded_proto = request.headers.get("x-forwarded-proto", "").strip().lower()
            if forwarded_proto == "https":
                return True

        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and attach security headers to the resulting response."""
        response = await call_next(request)

        if self.enabled:
            path = request.url.path
            is_docs = path.endswith(("/docs", "/redoc", "/openapi.json", "/ui")) or path in (
                "/",
                "/docs",
                "/redoc",
                "/ui",
            )
            for header_name, header_value in self.headers.items():
                # Do not overwrite if header is already explicitly set
                if header_name not in response.headers:
                    if header_name == "Content-Security-Policy" and is_docs:
                        response.headers[header_name] = DOCS_CSP
                    else:
                        response.headers[header_name] = header_value

            # Issue 4: Emit Strict-Transport-Security ONLY on true HTTPS or verified trusted proxy
            is_secure = self._is_secure(request)
            if self.enable_hsts and is_secure:
                if "Strict-Transport-Security" not in response.headers:
                    response.headers["Strict-Transport-Security"] = self.hsts_value
            elif not is_secure:
                # Never emit HSTS on plain HTTP; strip if attached
                if "Strict-Transport-Security" in response.headers:
                    del response.headers["Strict-Transport-Security"]

        return response
