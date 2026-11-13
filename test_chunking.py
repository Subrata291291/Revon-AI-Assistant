from pathlib import Path

import pytest

from chunker import CHUNK_OVERLAP, CHUNK_SIZE, create_chunks, normalize_text
from pdf_processor import extract_text_from_pdf


PDF_PATH = Path(
    r"C:\Users\ASUS\Desktop\Coding-ninja-Gen-AI-Course\Coding Ninjas Learning PDF\Pre-Read_ Basic Understanding on API  (1)-5983.pdf"
)


@pytest.fixture(scope="session")
def pdf_audit():
    if not PDF_PATH.is_file():
        pytest.skip(f"Test PDF not available: {PDF_PATH}")

    extracted_pages = extract_text_from_pdf(PDF_PATH.read_bytes())
    normalized_pages = [
        {
            "page_number": page["page_number"],
            "text": normalize_text(page["text"]),
        }
        for page in extracted_pages
    ]
    normalized_pages = [page for page in normalized_pages if page["text"]]
    chunks = create_chunks(normalized_pages, pages_are_normalized=True)

    return {
        "extracted_pages": extracted_pages,
        "normalized_pages": normalized_pages,
        "chunks": chunks,
    }


def _chunk_positions(source, chunks, page_number):
    positions = []
    previous_end = 0

    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        candidates = []
        search_from = 0
        while True:
            position = source.find(text, search_from)
            if position < 0:
                break
            candidates.append(position)
            search_from = position + 1

        if index == 0:
            aligned = [position for position in candidates if position == 0]
        else:
            aligned = [
                position
                for position in candidates
                if 40 <= previous_end - position <= 200
            ]

        if not aligned:
            expected_start = 0 if index == 0 else previous_end - CHUNK_OVERLAP
            low = max(0, expected_start - 80)
            high = min(len(source), expected_start + 81)
            best_position = expected_start
            best_prefix = -1
            for candidate_start in range(low, high):
                matched = 0
                while (
                    matched < len(text)
                    and candidate_start + matched < len(source)
                    and text[matched] == source[candidate_start + matched]
                ):
                    matched += 1
                if matched > best_prefix:
                    best_position = candidate_start
                    best_prefix = matched

            mismatch = best_position + max(best_prefix, 0)
            raise AssertionError(
                f"Page {page_number}, chunk {chunk['chunk_index']}: no exact "
                f"ordered source match; first mismatch near normalized page "
                f"offset {mismatch}. Source context: "
                f"{source[max(0, mismatch - 30):mismatch + 50]!r}; "
                f"chunk context: {text[max(0, best_prefix - 30):best_prefix + 50]!r}"
            )

        position = min(
            aligned,
            key=lambda candidate: (
                abs((previous_end - candidate) - CHUNK_OVERLAP),
                candidate,
            ),
        )
        positions.append(position)
        previous_end = position + len(text)

    return positions


def _audit_page(source, chunks, page_number):
    chunks = sorted(chunks, key=lambda chunk: chunk["chunk_index"])
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert chunks, f"Page {page_number} has no chunks."

    positions = _chunk_positions(source, chunks, page_number)
    assert positions[0] == 0, f"Page {page_number} chunks do not start at offset 0."
    assert positions[-1] + len(chunks[-1]["text"]) == len(source), (
        f"Page {page_number} chunks do not cover the source through its end."
    )
    return positions


# 1. PDF extraction validation
def test_pdf_extraction_validation(pdf_audit):
    pages = pdf_audit["extracted_pages"]
    assert pages
    assert all(page["text"].strip() for page in pages)
    assert [page["page_number"] for page in pages] == sorted(
        page["page_number"] for page in pages
    )


# 2. Text normalization validation
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  API   request\nwith\tspaces  ", "API request with spaces"),
        ("\u00a0TCP/IP\u200b and JSON\ufeff", "TCP/IP and JSON"),
        ("parts\u200band", "parts and"),
        ("Keep this space . Keep this URL: https://api.example.com/v1.2", (
            "Keep this space . Keep this URL: https://api.example.com/v1.2"
        )),
    ],
)
def test_text_normalization(raw, expected):
    assert normalize_text(raw) == expected


def test_pdf_page_normalization(pdf_audit):
    normalized_pages = pdf_audit["normalized_pages"]
    assert normalized_pages
    assert all(page["text"] == page["text"].strip() for page in normalized_pages)
    assert all(
        not any(
            page["text"][index].isspace()
            and page["text"][index + 1].isspace()
            for index in range(len(page["text"]) - 1)
        )
        for page in normalized_pages
    )


# 3. Exact chunk preservation
def test_pdf_chunks_are_exact_source_substrings(pdf_audit):
    sources = {
        page["page_number"]: page["text"]
        for page in pdf_audit["normalized_pages"]
    }
    chunks_by_page = {page_number: [] for page_number in sources}
    for chunk in pdf_audit["chunks"]:
        chunks_by_page[chunk["page_number"]].append(chunk)

    for page_number, source in sources.items():
        _audit_page(source, chunks_by_page[page_number], page_number)


def test_repeated_phrases_align_in_source_order():
    raw = "Echo phrase appears again. " * 100
    source = normalize_text(raw)
    pages = [{"page_number": 7, "text": source}]
    chunks = create_chunks(pages, pages_are_normalized=True)

    positions = _audit_page(source, chunks, page_number=7)
    assert positions == sorted(positions)


# 4. Chunk size validation
def test_pdf_chunks_respect_size_limit(pdf_audit):
    oversized = [
        (chunk["page_number"], chunk["chunk_index"], len(chunk["text"]))
        for chunk in pdf_audit["chunks"]
        if len(chunk["text"]) > CHUNK_SIZE
    ]
    assert not oversized, f"Oversized chunks (page, index, length): {oversized}"


def test_synthetic_long_text_respects_size_limit():
    source = normalize_text("A technical paragraph about APIs. " * 100)
    chunks = create_chunks(
        [{"page_number": 1, "text": source}],
        pages_are_normalized=True,
    )
    assert len(source) > CHUNK_SIZE
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)


def test_default_create_chunks_interface_normalizes_and_keeps_metadata():
    raw = "The API response contains JSON and an HTTP status code.  \n" * 40
    source = normalize_text(raw)
    chunks = create_chunks([{"page_number": 9, "text": raw}])

    assert len(source) > CHUNK_SIZE
    assert all(
        set(chunk) == {"page_number", "chunk_index", "text"}
        and chunk["page_number"] == 9
        for chunk in chunks
    )
    _audit_page(source, chunks, page_number=9)


# 5. Overlap validation
def test_pdf_chunks_have_approximately_120_character_overlap(pdf_audit):
    sources = {
        page["page_number"]: page["text"]
        for page in pdf_audit["normalized_pages"]
    }
    chunks_by_page = {page_number: [] for page_number in sources}
    for chunk in pdf_audit["chunks"]:
        chunks_by_page[chunk["page_number"]].append(chunk)

    for page_number, page_chunks in chunks_by_page.items():
        page_chunks.sort(key=lambda chunk: chunk["chunk_index"])
        positions = _chunk_positions(sources[page_number], page_chunks, page_number)
        for previous, current, previous_start, current_start in zip(
            page_chunks, page_chunks[1:], positions, positions[1:]
        ):
            overlap = previous_start + len(previous["text"]) - current_start
            assert 40 <= overlap <= 200, (
                f"Page {page_number}, chunk {current['chunk_index']}: "
                f"overlap is {overlap}, expected approximately {CHUNK_OVERLAP}."
            )


# 6. Sentence and word boundary validation
def test_sentence_boundaries_are_preferred():
    source = normalize_text(
        "Each API request has a clear sentence boundary. " * 50
    )
    chunks = create_chunks(
        [{"page_number": 1, "text": source}],
        pages_are_normalized=True,
    )
    positions = _audit_page(source, chunks, page_number=1)

    for chunk, start, following in zip(chunks, positions, positions[1:]):
        end = start + len(chunk["text"])
        assert source[end - 1] in ".!?"
        assert source[end].isspace()
        assert following > start
    assert all(
        chunk["text"][0] not in ".,;:!?"
        for chunk in chunks[1:]
    )


def test_word_boundaries_are_used_without_splitting_tokens():
    source = " ".join(f"token{index}" for index in range(250))
    chunks = create_chunks(
        [{"page_number": 1, "text": source}],
        pages_are_normalized=True,
    )
    positions = _audit_page(source, chunks, page_number=1)

    for index, (chunk, start) in enumerate(zip(chunks, positions)):
        end = start + len(chunk["text"])
        if index < len(chunks) - 1:
            assert source[end].isspace()
        if index > 0:
            assert source[start - 1].isspace()


# 7. Technical terminology preservation
def test_long_text_preserves_punctuation_urls_and_technical_terms():
    paragraph = (
        "Use https://api.example.com/v1.2 with OAuth 2.0, JSON, TCP/IP, "
        "C++, .NET, and TLS; e.g. keep API names unchanged.  "
        "A second line tests newlines and  multiple spaces!\n"
    )
    raw = paragraph * 30
    source = normalize_text(raw)
    chunks = create_chunks(
        [{"page_number": 3, "text": source}],
        pages_are_normalized=True,
    )
    assert len(source) > CHUNK_SIZE
    _audit_page(source, chunks, page_number=3)

    for term in (
        "https://api.example.com/v1.2",
        "OAuth 2.0",
        "JSON",
        "TCP/IP",
        "C++",
        ".NET",
        "TLS",
        "e.g.",
    ):
        assert any(term in chunk["text"] for chunk in chunks), term
