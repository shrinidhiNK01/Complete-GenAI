"""
Advanced RAG Pipeline with Query Expansion and Re-ranking.
This script implements:
1. Document chunking with character overlap.
2. Generating embeddings using SentenceTransformers.
3. Indexing documents in our custom Vector Store.
4. Query Expansion: Generating query variations.
5. Re-ranking: Scoring retrieved candidates using a Cross-Encoder.
6. Context assembly.

Run: python Complete_Course/RAG/implementations/advanced_rag_pipeline.py
"""

import os
from vector_store_from_scratch import ScratchVectorStore

# Try importing models, define fallback mock behavior if libraries aren't installed
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

class AdvancedRAGPipeline:
    def __init__(self):
        self.vector_store = ScratchVectorStore()
        if MODELS_AVAILABLE:
            print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loading CrossEncoder re-ranker model ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            print("[WARNING] sentence-transformers not installed. Running in mock mode.")
            self.embedder = None
            self.reranker = None

    def chunk_document(self, text, chunk_size=300, overlap=50):
        """Splits long text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_embeddings(self, texts):
        """Generates vectors for a list of texts."""
        if MODELS_AVAILABLE:
            return self.embedder.encode(texts, convert_to_numpy=True)
        else:
            # Mock embeddings: generate random 384-dimensional vectors
            import numpy as np
            np.random.seed(42)
            return np.random.randn(len(texts), 384)

    def ingest_document(self, doc_text, metadata=None):
        """Chunk a document and add it to the vector database."""
        chunks = self.chunk_document(doc_text)
        embeddings = self.get_embeddings(chunks)
        self.vector_store.add_documents(chunks, embeddings, [metadata for _ in chunks])
        print(f"Ingested {len(chunks)} chunks into the vector store.")

    def query_expansion(self, original_query):
        """Simulate generating alternative query forms to broaden vector matching."""
        # E.g., adding synonyms or phrasing options
        # In production, you would prompt an LLM: "Generate 2 synonyms for: [query]"
        variations = [original_query]
        if "refund" in original_query.lower():
            variations.append("money back policy")
            variations.append("return payment details")
        elif "shipping" in original_query.lower() or "delivery" in original_query.lower():
            variations.append("shipping times cost")
            variations.append("package transit duration")
        return variations

    def retrieve_and_rerank(self, query, k_vector=10, k_final=3):
        """
        Step 1: Retrieve candidate chunks via vector similarity (Query Expansion included).
        Step 2: Re-rank retrieved candidates using a Cross-Encoder.
        """
        queries = self.query_expansion(query)
        print(f"Expanded queries: {queries}")

        # Gather candidates from all expanded queries
        all_candidates = {}
        for q in queries:
            q_emb = self.get_embeddings([q])[0]
            matches = self.vector_store.search(q_emb, k=k_vector)
            for m in matches:
                # Deduplicate by document text
                all_candidates[m["document"]] = m

        candidate_list = list(all_candidates.values())
        print(f"Retrieved {len(candidate_list)} unique candidate chunks from vector search.")

        # Re-rank candidates
        if MODELS_AVAILABLE and len(candidate_list) > 0:
            # Format inputs as pairs: [[query, doc1], [query, doc2], ...]
            pairs = [[query, c["document"]] for c in candidate_list]
            scores = self.reranker.predict(pairs)
            
            # Update scores with Cross-Encoder ratings
            for idx, score in enumerate(scores):
                candidate_list[idx]["score"] = float(score)
            
            # Sort by new score
            candidate_list.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Fallback/mock re-ranking: keep vector score sorting
            candidate_list.sort(key=lambda x: x["score"], reverse=True)

        return candidate_list[:k_final]


def main():
    pipeline = AdvancedRAGPipeline()

    # Sample Document
    company_manual = (
        "Refund Policy: Customers can request a full refund on any product within 30 days of purchase. "
        "To qualify, the item must be returned unused in its original box packaging. "
        "Refunds are processed to the original credit card within 5-10 business days. "
        "Shipping Details: We offer flat-rate standard shipping of 5 dollars on all orders under 50 dollars. "
        "Orders over 50 dollars qualify for free standard shipping. "
        "Standard shipping takes 3 to 5 business days. "
        "Express shipping is available for 15 dollars and takes 1 to 2 business days. "
        "Warranty Rules: All hardware products include a 1-year limited warranty covering manufacturer defects. "
        "It does not cover accidental water damage or screen cracks."
    )

    # Ingest document
    pipeline.ingest_document(company_manual, metadata={"source": "faq_manual"})

    # Run Query
    query = "How long do returns take to get my money back?"
    print(f"\n=== Running RAG Query: '{query}' ===")
    
    results = pipeline.retrieve_and_rerank(query, k_vector=5, k_final=2)
    
    print("\n=== Top Re-ranked Retrieved Context ===")
    for i, r in enumerate(results):
        print(f"Rank {i+1} [Score: {r['score']:.4f}]: {r['document']}\n")

    # Construct the final prompt to feed to a generation model
    context = "\n".join([f"- {r['document']}" for r in results])
    prompt = (
        "Instructions: Answer the question using ONLY the context provided below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    print("=== Compiled Generation Prompt ===")
    print(prompt)

if __name__ == '__main__':
    main()
