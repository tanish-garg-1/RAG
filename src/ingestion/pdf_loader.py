from collections.abc import Iterator

import pymupdf

from config import settings
from src.models import PageImage

MAX_IMAGE_SIDE = 2000  # larger images are shrunk before upload; vision models downscale them anyway


class PDFLoader:
    def load(self, pdf_path: str) -> list[tuple[int, str]]:
        pages = []
        doc = pymupdf.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                print(f"Warning: page {page_number} has no extractable text (possibly scanned/image-only).")
            pages.append((page_number, text))

        doc.close()
        return pages

    def iter_images(self, pdf_path: str) -> Iterator[PageImage]:
        """Yields what a vision model needs to see, one item at a time (so a large PDF is never held in memory).

        - A page with no extractable text is rendered whole: it is scanned, and its text must be transcribed.
        - Any other page yields each embedded image at least `min_image_px` on both sides.
        """
        doc = pymupdf.open(pdf_path)
        try:
            for page_number, page in enumerate(doc, start=1):
                if not page.get_text().strip():
                    pixmap = page.get_pixmap(dpi=settings.page_render_dpi)
                    yield PageImage(page_number, 0, pixmap.tobytes("png"), "image/png", is_full_page=True)
                    continue

                seen: set[int] = set()
                index = 0
                for info in page.get_images(full=True):
                    xref, width, height = info[0], info[2], info[3]
                    if xref in seen or min(width, height) < settings.min_image_px:
                        continue
                    seen.add(xref)
                    data = self._to_png(doc, xref)
                    if data:
                        yield PageImage(page_number, index, data, "image/png", is_full_page=False)
                        index += 1
        finally:
            doc.close()

    @staticmethod
    def _to_png(doc, xref: int) -> bytes | None:
        try:
            pixmap = pymupdf.Pixmap(doc, xref)
            if pixmap.n - pixmap.alpha >= 4:   # CMYK cannot be saved as PNG
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
            shrink = 0
            while max(pixmap.width, pixmap.height) / (2 ** shrink) > MAX_IMAGE_SIDE:
                shrink += 1
            if shrink:
                pixmap.shrink(shrink)
            return pixmap.tobytes("png")
        except (RuntimeError, ValueError):
            return None   # unsupported or corrupt image: skip it rather than fail the whole ingest
