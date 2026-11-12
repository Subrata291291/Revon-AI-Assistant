
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# Initialize official LangChain splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


def create_chunks(extracted_pages):
    """
    Create chunks from extracted PDF pages.

    Each chunk preserves:
    - page_number
    - chunk_index
    - text
    """

    all_chunks = []

    for page in extracted_pages:

        page_number = page["page_number"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        # Split page text using LangChain
        page_chunks = text_splitter.split_text(
            page_text
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks
        ):

            all_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text
            })

    return all_chunks
