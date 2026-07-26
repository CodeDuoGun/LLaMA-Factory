"""Small dependency-free checks for complete common raster image containers."""

from __future__ import annotations

from pathlib import Path


def is_complete_image(path: Path, *, decode_pixels: bool = False) -> bool:
    try:
        size = path.stat().st_size
        if size < 32:
            return False
        with path.open("rb") as handle:
            head = handle.read(16)
            handle.seek(max(0, size - 256))
            tail = handle.read(256)
        if head.startswith(b"\xff\xd8"):
            # Some valid exporters append a small trailer after the JPEG EOI.
            container_complete = b"\xff\xd9" in tail
        elif head.startswith(b"\x89PNG\r\n\x1a\n"):
            container_complete = b"IEND\xaeB`\x82" in tail
        elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            container_complete = int.from_bytes(head[4:8], "little") + 8 == size
        elif head.startswith(b"BM"):
            container_complete = int.from_bytes(head[2:6], "little") <= size
        else:
            container_complete = False
        if not container_complete or not decode_pixels:
            return container_complete

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            image.load()
        return True
    except (ImportError, OSError, ValueError):
        return False
