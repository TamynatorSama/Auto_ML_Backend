class SandboxError(Exception):
    status = 500


class BadRequest(SandboxError):
    status = 400


class NotFound(SandboxError):
    status = 404


class Conflict(SandboxError):
    status = 409


class TooLarge(SandboxError):
    status = 413


class NoCapacity(SandboxError):
    status = 429
