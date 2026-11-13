
import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


_SENTENCE_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "vs.",
    "etc.",
)
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?](?=\s)")
_WHITESPACE_PATTERN = re.compile(r"\s")


def normalize_text(text: str) -> str:
    """Normalize extracted whitespace without guessing missing word breaks."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_sentence_boundary(text: str, position: int) -> bool:
    punctuation = text[position - 1]
    if punctuation != ".":
        return True

    preceding_text = text[max(0, position - 8):position].lower()
    return not any(
        preceding_text.endswith(abbreviation)
        for abbreviation in _SENTENCE_ABBREVIATIONS
    )


def _find_chunk_end(text: str, start: int) -> int:
    limit = min(start + CHUNK_SIZE, len(text))
    if limit == len(text):
        return limit

    minimum = start + CHUNK_SIZE // 2
    sentence_ends = []
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text, start + 1, limit):
        if match.end() >= minimum and _is_sentence_boundary(text, match.end()):
            sentence_ends.append(match.end())
    if sentence_ends:
        return sentence_ends[-1]

    word_ends = [
        match.start()
        for match in _WHITESPACE_PATTERN.finditer(text, minimum, limit)
    ]
    return word_ends[-1] if word_ends else limit


def _find_next_chunk_start(
    text: str,
    previous_start: int,
    previous_end: int,
) -> int:
    target = previous_end - CHUNK_OVERLAP
    minimum = max(previous_start + 1, previous_end - 200)
    maximum = previous_end - 40

    word_starts = [
        position
        for position in range(minimum, maximum + 1)
        if text[position - 1].isspace()
        and text[position] not in ".,;:!?"
    ]
    if word_starts:
        return min(
            word_starts,
            key=lambda position: (abs(position - target), -position),
        )

    return max(previous_start + 1, target)


def _split_text(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0

    while start < len(text):
        end = _find_chunk_end(text, start)
        spans.append((start, end))
        if end == len(text):
            break

        next_start = _find_next_chunk_start(text, start, end)
        if next_start <= start:
            raise RuntimeError("Chunking failed to advance through the source text.")
        start = next_start

    return spans


def create_chunks(extracted_pages, *, pages_are_normalized: bool = False):
    """
    Create chunks from extracted PDF pages.

    Each chunk preserves:
    - page_number
    - chunk_index
    - text

    Set pages_are_normalized when the supplied page text has already passed
    through normalize_text.
    """

    all_chunks = []

    for page in extracted_pages:
        page_number = page["page_number"]
        page_text = (
            page["text"]
            if pages_are_normalized
            else normalize_text(page["text"])
        )
        if not page_text:
            continue

        for chunk_index, (start, end) in enumerate(_split_text(page_text)):
            chunk_text = page_text[start:end]
            if not chunk_text.strip():
                continue

            all_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text
            })

    return all_chunks
