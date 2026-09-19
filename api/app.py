"""
The API (docs/PHASE3.md):

    .venv\\Scripts\\python.exe -m uvicorn api.app:app --reload --reload-dir api --reload-dir db

Needs DATABASE_URL in .env and the tunnel to the server. The web app's dev
server forwards /api here, so the browser sees one origin.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

import db
from api import auth, config, oauth, tenancy, workspaces

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def create_app(pool=None) -> FastAPI:
    """pool: the tests' own; without one the app opens (and later closes) a pool on DATABASE_URL."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = pool or db.pool(size=10)
        with app.state.pool.connection() as conn:
            db.migrate(conn)
        yield
        if pool is None:
            app.state.pool.close()

    app = FastAPI(title="automl", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        # on top of SameSite=Lax: anything that changes something must come from the web app itself
        if request.method not in SAFE_METHODS and request.headers.get("origin") != config.APP_ORIGIN:
            return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"     # no guessing an answer is HTML
        response.headers["X-Frame-Options"] = "DENY"               # nothing here belongs in a frame
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # only the OAuth redirect and callback use it (api/oauth.py)
    app.add_middleware(SessionMiddleware, secret_key=config.oauth_state_secret(), session_cookie="automl_oauth",
                       max_age=600, path="/api/auth/oauth", same_site="lax", https_only=config.SECURE)
    app.include_router(auth.router)
    app.include_router(oauth.router)
    app.include_router(workspaces.router)
    app.include_router(tenancy.router)
    return app


app = create_app()
