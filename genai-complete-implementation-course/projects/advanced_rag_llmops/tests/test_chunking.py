from app.ingestion.chunking import chunk_document
from app.ingestion.models import Document


def test_chunk_document_with_overlap():
    document = Document(doc_id="doc", text="one two three four five six")
    chunks = chunk_document(document, chunk_size_words=4, overlap_words=2)
    assert len(chunks) == 2
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"

