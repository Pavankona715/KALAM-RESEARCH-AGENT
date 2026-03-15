"""
Parsers Package
===============
Routes file types to the correct parser.

Usage:
    from backend.ingestion.parsers import parse_document
    text = await parse_document("/uploads/report.pdf")
    text = await parse_document("https://example.com/page")
"""

from backend.ingestion.parsers.url_parser import parse_url, parse_html_file

async def parse_document(source, *, is_url: bool = False) -> str:
    """Parse any supported document into plain text."""
    from pathlib import Path

    if is_url or (isinstance(source, str) and source.startswith(("http://", "https://"))):
        from backend.ingestion.parsers.url_parser import parse_url
        return await parse_url(str(source))

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        from backend.ingestion.parsers.pdf_parser import parse_pdf
        return await parse_pdf(path)
    elif ext in (".docx", ".doc"):
        from backend.ingestion.parsers.docx_parser import parse_docx
        return await parse_docx(path)
    elif ext in (".txt", ".md", ".markdown"):
        from backend.ingestion.parsers.txt_parser import parse_txt
        return await parse_txt(path)
    elif ext in (".html", ".htm"):
        from backend.ingestion.parsers.url_parser import parse_html_file
        return await parse_html_file(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Supported: .pdf .docx .txt .md .html"
        )

__all__ = ["parse_document", "parse_url", "parse_html_file"]