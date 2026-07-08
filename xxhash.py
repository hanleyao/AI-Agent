from __future__ import annotations

import hashlib


class _XXH3_128:
    def __init__(self, data: bytes | bytearray | memoryview | str = b"") -> None:
        if isinstance(data, str):
            data = data.encode()
        self._hash = hashlib.blake2b(bytes(data), digest_size=16)

    def update(self, data: bytes | bytearray | memoryview | str) -> None:
        if isinstance(data, str):
            data = data.encode()
        self._hash.update(bytes(data))

    def digest(self) -> bytes:
        return self._hash.digest()

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def xxh3_128(data: bytes | bytearray | memoryview | str = b"") -> _XXH3_128:
    return _XXH3_128(data)


def xxh3_128_hexdigest(data: bytes | bytearray | memoryview | str = b"") -> str:
    return xxh3_128(data).hexdigest()
