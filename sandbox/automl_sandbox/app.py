"""HTTP routes and the token check.

    uvicorn --factory automl_sandbox.app:from_config --host 0.0.0.0 --port 8765
"""

import hmac
import logging
import os
import re
import threading
from pathlib import PurePosixPath
from typing import Annotated, Optional

import docker.errors
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from automl_sandbox import __version__, config as config_module, tarsafe
from automl_sandbox.config import Config
from automl_sandbox.errors import BadRequest, NoCapacity, SandboxError, TooLarge

ID = r"^[0-9a-f]{12}$"
LABEL_KEY = r"^[A-Za-z0-9_.-]{1,63}$"

ObjectId = Annotated[str, PathParam(pattern=ID)]
Labels = Annotated[
    dict[Annotated[str, StringConstraints(pattern=LABEL_KEY)], Annotated[str, StringConstraints(max_length=256)]],
    Field(max_length=32),
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VolumeIn(Strict):
    labels: Labels = {}
    ttl_seconds: Optional[int] = Field(None, gt=0)


class SandboxIn(Strict):
    image: str
    cpus: float = Field(gt=0)
    memory_mb: int = Field(gt=0)
    work_mb: int = Field(gt=0)
    ttl_seconds: Optional[int] = Field(None, gt=0)
    volume: Optional[str] = Field(None, pattern=ID)
    labels: Labels = {}


class ExecIn(Strict):
    cmd: list[str] = Field(min_length=1)
    deadline_seconds: float = Field(gt=0)


def at_most(name: str, value, limit) -> None:
    if value > limit:
        raise BadRequest(f"{name} {value} is over the maximum {limit}")


def create_app(config: Config, engine) -> FastAPI:
    app = FastAPI(title="automl-sandbox", version=__version__)
    expected = f"Bearer {config.token}".encode()

    def check_token(authorization: str = Header("")):
        if not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(401, "missing or wrong token")

    def ttl(seconds: Optional[int]) -> int:
        if seconds is None:
            return config.default_ttl_seconds
        at_most("ttl_seconds", seconds, config.max_ttl_seconds)
        return seconds

    async def read_archive(request: Request):
        limit = config.max_upload_mb * 1024 * 1024
        too_large = TooLarge(f"upload is over {config.max_upload_mb} MB")
        if int(request.headers.get("content-length") or 0) > limit:
            raise too_large
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                raise too_large
        return await run_in_threadpool(tarsafe.clean, bytes(body), config.sandbox_user)

    @app.exception_handler(SandboxError)
    async def sandbox_error(request, error: SandboxError):
        headers = {"Retry-After": "5"} if isinstance(error, NoCapacity) else None
        return JSONResponse({"detail": str(error)}, error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error: RequestValidationError):
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return JSONResponse({"detail": detail}, 400)

    @app.exception_handler(docker.errors.DockerException)
    async def docker_error(request, error: docker.errors.DockerException):
        status = error.status_code if isinstance(error, docker.errors.APIError) else None
        explanation = getattr(error, "explanation", None) or str(error)
        return JSONResponse({"detail": f"docker: {explanation}"}, status if status in (404, 409) else 502)

    @app.get("/v1/health")
    def health():
        return {"ok": True}

    api = APIRouter(prefix="/v1", dependencies=[Depends(check_token)])

    @api.get("/info")
    def info():
        return engine.info()

    @api.get("/capacity")
    def capacity():
        return engine.capacity()

    @api.post("/volumes", status_code=201)
    def create_volume(body: VolumeIn):
        return engine.create_volume(body.labels, ttl(body.ttl_seconds))

    @api.put("/volumes/{volume_id}/files")
    async def put_volume_files(volume_id: ObjectId, request: Request):
        archive, files, size = await read_archive(request)
        await run_in_threadpool(engine.put_volume_files, volume_id, archive)
        return {"files": files, "bytes": size}

    @api.delete("/volumes/{volume_id}")
    def delete_volume(volume_id: ObjectId):
        engine.delete_volume(volume_id)
        return {"deleted": volume_id}

    @api.post("/sandboxes", status_code=201)
    def create_sandbox(body: SandboxIn):
        if body.image not in config.images:
            raise BadRequest(f"unknown image {body.image!r}; allowed: {', '.join(config.images)}")
        at_most("cpus", body.cpus, config.max_cpus)
        at_most("memory_mb", body.memory_mb, config.max_sandbox_memory_mb)
        at_most("work_mb", body.work_mb, config.max_work_mb)
        return engine.create_sandbox(body.image, body.cpus, body.memory_mb, body.work_mb,
                                     ttl(body.ttl_seconds), body.volume, body.labels)

    @api.get("/sandboxes")
    def list_sandboxes(label: list[str] = Query([])):
        pairs = []
        for item in label:
            key, sep, value = item.partition(":")
            if not sep or not re.match(LABEL_KEY, key):
                raise BadRequest(f"label filter must be key:value, got {item!r}")
            pairs.append((key, value))
        return engine.list_sandboxes(pairs)

    @api.put("/sandboxes/{sandbox_id}/files")
    async def put_files(sandbox_id: ObjectId, request: Request):
        archive, files, size = await read_archive(request)
        await run_in_threadpool(engine.put_files, sandbox_id, archive)
        return {"files": files, "bytes": size}

    @api.get("/sandboxes/{sandbox_id}/files")
    def get_files(sandbox_id: ObjectId, path: str):
        clean = PurePosixPath(path)
        if clean.is_absolute() or ".." in clean.parts:
            raise BadRequest("path must be relative and stay inside /work")
        return StreamingResponse(engine.get_files(sandbox_id, str(clean)), media_type="application/x-tar")

    @api.post("/sandboxes/{sandbox_id}/kill")
    def kill(sandbox_id: ObjectId):
        engine.kill(sandbox_id)
        return {"killed": sandbox_id}

    @api.delete("/sandboxes/{sandbox_id}")
    def delete_sandbox(sandbox_id: ObjectId):
        return {"deleted": sandbox_id, "usage": engine.delete_sandbox(sandbox_id)}

    @api.post("/sandboxes/{sandbox_id}/exec", status_code=202)
    def start_exec(sandbox_id: ObjectId, body: ExecIn):
        at_most("deadline_seconds", body.deadline_seconds, config.max_exec_seconds)
        return {"exec_id": engine.start_exec(sandbox_id, body.cmd, body.deadline_seconds)}

    @api.get("/sandboxes/{sandbox_id}/exec/{exec_id}")
    def get_exec(sandbox_id: ObjectId, exec_id: ObjectId, stdout_offset: int = 0, stderr_offset: int = 0):
        return engine.get_exec(sandbox_id, exec_id, stdout_offset, stderr_offset)

    app.include_router(api)
    return app


def from_config() -> FastAPI:
    from automl_sandbox.engine import Engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = config_module.load(os.environ.get("AUTOML_SANDBOX_CONFIG", "/etc/automl-sandbox/config.toml"))
    engine = Engine(config)
    threading.Thread(target=engine.reap_forever, daemon=True).start()
    return create_app(config, engine)
