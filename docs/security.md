# Security model

The private workbench is single-admin software, not a multi-tenant identity
platform.

- Browser login verifies `API_ADMIN_PASSWORD_HASH` with Argon2.
- The session payload is HMAC signed, expires after eight hours, and is stored
  in a `HttpOnly`, `SameSite=Lax` cookie.
- Mutating cookie-authenticated requests require the signed CSRF token in both
  the readable cookie and `X-CSRF-Token` header.
- CLI clients use `X-API-Key` or a Bearer header. Credentials in query strings
  are rejected by design.
- API keys are hashed before becoming rate-limit identifiers and are not logged.
- Login attempts have a separate, lower rate limit. Forwarded client IP headers
  are ignored unless `TRUST_PROXY_HEADERS=1` is explicitly configured behind a
  proxy that overwrites untrusted values.
- Generic Agent tool execution and file tools are separately disabled by
  default. File operations are restricted to configured output roots.
- Production Compose does not publish Postgres, Redis, or MinIO ports.

Rotate `API_SECRET_KEY` and `API_SESSION_SECRET` independently. Rotating the
session secret logs out existing browsers. OpenAPI and interactive docs require
authentication when API auth is enabled; set `PUBLIC_API_DOCS=1` only for an
intentionally public API.

Known limitation: the current signed session has no server-side revocation list;
logout clears the browser cookie, while forced revocation requires secret
rotation. This is acceptable for the selected single-admin boundary but should
be replaced before multi-user support.
