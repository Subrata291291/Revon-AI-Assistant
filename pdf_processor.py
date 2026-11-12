
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes):

    # Open PDF from memory
    pdf_file = BytesIO(file_bytes)

    pdf_reader = PdfReader(pdf_file)

    extracted_pages = []

    # Extract text page by page
    for page_number, page in enumerate(pdf_reader.pages, start=1):

        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return extracted_pages
