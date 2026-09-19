"""Keeps the newest bytes of one output stream, addressed by absolute offset."""

import threading


class OutputBuffer:
    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()
        self.start = 0  # absolute offset of data[0]
        self.lock = threading.Lock()

    def write(self, chunk: bytes) -> None:
        with self.lock:
            self.data += chunk
            extra = len(self.data) - self.limit
            if extra > 0:
                del self.data[:extra]
                self.start += extra

    def read(self, offset: int):
        """Bytes from offset on -> (text, next offset, whether older bytes were dropped)."""
        with self.lock:
            end = self.start + len(self.data)
            offset = min(max(offset, 0), end)
            skipped = offset < self.start
            begin = max(offset, self.start) - self.start
            return self.data[begin:].decode("utf-8", errors="replace"), end, skipped
