"""
oauth.py
--------
Google and GitHub sign-in through Authlib (docs/PHASE3.md §4). Between the
redirect and the callback, the state, PKCE verifier and Google's nonce wait in
a signed ten-minute cookie, `automl_oauth` (SessionMiddleware in app.py).

- an identity already linked: that user
- else an account with the same email: linked, only if the provider says the email is verified
- else a new user, if the email is on the allowlist; the Welcome screen then asks for a workspace

Needs GOOGLE_OAUTH_CLIENT_ID / _SECRET and GITHUB_OAUTH_CLIENT_ID / _SECRET in
.env; a provider without them answers 404.
"""

import os
from urllib.parse import quote

import httpx
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from api import config, sessions
from api.auth import NOT_INVITED, invited
from api.workspaces import SLUG

router = APIRouter(prefix="/api/auth/oauth")
oauth = OAuth()
NAMES = {"google": "Google", "github": "GitHub"}
PROVIDERS = {
    "google": {"server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
               "client_kwargs": {"scope": "openid email profile", "code_challenge_method": "S256"}},
    "github": {"authorize_url": "https://github.com/login/oauth/authorize",
               "access_token_url": "https://github.com/login/oauth/access_token",
               "api_base_url": "https://api.github.com/",
               "client_kwargs": {"scope": "read:user user:email", "code_challenge_method": "S256"}},
}
for _name, _settings in PROVIDERS.items():
    if os.environ.get(f"{_name.upper()}_OAUTH_CLIENT_ID"):
        oauth.register(_name, client_id=os.environ[f"{_name.upper()}_OAUTH_CLIENT_ID"],
                       client_secret=os.environ.get(f"{_name.upper()}_OAUTH_CLIENT_SECRET"), **_settings)


@router.get("/providers")
def enabled_providers():
    return {"providers": [name for name in PROVIDERS if oauth.create_client(name) is not None]}


class Refused(Exception):
    """Sign-in didn't happen; the message is for the log-in screen."""


def _client(provider: str):
    client = oauth.create_client(provider) if provider in PROVIDERS else None
    if client is None:
        raise HTTPException(404, "Not found")
    return client


async def profile(provider: str, request: Request) -> dict:
    """Finishes the exchange with the provider: {subject, email, verified, name}."""
    client = _client(provider)
    token = await client.authorize_access_token(request)
    if provider == "google":
        info = token["userinfo"]   # from the ID token, which Authlib has checked
        return {"subject": info["sub"], "email": info.get("email"), "verified": info.get("email_verified") is True,
                "name": info.get("name")}
    user = await client.get("user", token=token)
    emails = await client.get("user/emails", token=token)
    user.raise_for_status()
    emails.raise_for_status()
    primary = next((entry for entry in emails.json() if entry.get("primary")), {})
    return {"subject": str(user.json()["id"]), "email": primary.get("email"), "verified": primary.get("verified") is True,
            "name": user.json().get("name") or user.json().get("login")}


def sign_in(pool, provider: str, person: dict, response: Response, request: Request) -> str:
    """Finds, links or makes the user and signs them in; returns where the web app goes next."""
    email = (person["email"] or "").strip().lower()
    with pool.connection() as conn, conn.transaction():
        linked = conn.execute("SELECT user_id FROM oauth_identities WHERE provider = %s AND subject = %s",
                              (provider, person["subject"])).fetchone()
        if linked:
            user_id = linked["user_id"]
        else:
            if not email or not person["verified"]:
                raise Refused(f"Your {NAMES[provider]} account has no verified email address")
            user = conn.execute("SELECT id, email_verified_at FROM users WHERE email = %s FOR UPDATE",
                                (email,)).fetchone()
            if user:
                user_id = user["id"]
                if user["email_verified_at"] is None:
                    # a password sign-up still waiting for its link may not be this person's:
                    # its password, workspace and links go
                    conn.execute("UPDATE users SET email_verified_at = now(), password_hash = NULL WHERE id = %s",
                                 (user_id,))
                    conn.execute("DELETE FROM workspaces WHERE id IN "
                                 "(SELECT workspace_id FROM memberships WHERE user_id = %s)", (user_id,))
                    conn.execute("UPDATE email_tokens SET used_at = now() WHERE user_id = %s AND used_at IS NULL",
                                 (user_id,))
            elif invited(conn, email):
                user_id = conn.execute(
                    "INSERT INTO users (email, name, email_verified_at) VALUES (%s, %s, now()) RETURNING id",
                    (email, person["name"] or email.split("@")[0]),
                ).fetchone()["id"]
            else:
                raise Refused(NOT_INVITED)
            conn.execute("INSERT INTO oauth_identities (user_id, provider, subject, email) VALUES (%s, %s, %s, %s)",
                         (user_id, provider, person["subject"], email))
        sessions.sign_in(conn, response, user_id, request)
        workspace = conn.execute("SELECT w.slug FROM memberships m JOIN workspaces w ON w.id = m.workspace_id "
                                 "WHERE m.user_id = %s ORDER BY m.created_at LIMIT 1", (user_id,)).fetchone()
    # only a slug-shaped path: a workspace made by the command line could hold anything,
    # and "//elsewhere.example" in a Location header leaves the site
    if workspace and SLUG.fullmatch(workspace["slug"]):
        return f"/{workspace['slug']}"
    return "/welcome"


@router.get("/{provider}")
async def start(provider: str, request: Request):
    # the callback goes through the web app's origin (the Vite proxy in development)
    return await _client(provider).authorize_redirect(request, f"{config.APP_ORIGIN}/api/auth/oauth/{provider}/callback")


@router.get("/{provider}/callback")
async def callback(provider: str, request: Request):
    _client(provider)
    response = Response(status_code=303)
    try:
        try:
            person = await profile(provider, request)
        except (OAuthError, httpx.HTTPError, KeyError):
            raise Refused(f"Signing in with {NAMES[provider]} didn't finish. Try again.")
        response.headers["location"] = await run_in_threadpool(sign_in, request.app.state.pool, provider, person,
                                                               response, request)
    except Refused as refusal:
        return Response(status_code=303, headers={"location": f"/login?error={quote(str(refusal))}"})
    return response
