from app.ingestion.models import Chunk
from app.rag.retrievers import HybridRetriever
from app.rag.vector_store import LocalTfidfIndex


def test_hybrid_retrieval_finds_refund_chunk():
    chunks = [
        Chunk("c1", "policy", "Refunds are allowed within 30 days.", 0, 6, {"file_name": "refund.txt"}),
        Chunk("c2", "shipping", "Shipping takes 3 to 5 business days.", 0, 7, {"file_name": "shipping.txt"}),
    ]
    index = LocalTfidfIndex.build(chunks)
    retriever = HybridRetriever(index)
    results = retriever.search("refund window", top_k=4, rerank_top_k=1)
    assert results[0].chunk.chunk_id == "c1"

