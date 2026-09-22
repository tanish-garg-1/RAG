import pymupdf


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
