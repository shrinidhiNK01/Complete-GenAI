"""
Vector Store built from Scratch using NumPy.
This script demonstrates:
1. Embedding storage.
2. Cosine similarity matching using NumPy.
3. Metadata filtering prior to vector similarity calculations.
4. Serializing (saving/loading) the database index.

Run: python Complete_Course/RAG/implementations/vector_store_from_scratch.py
"""

import json
import numpy as np

class ScratchVectorStore:
    def __init__(self):
        self.vectors = []      # List of np.ndarray
        self.documents = []    # List of strings (original text chunks)
        self.metadata = []     # List of dicts (associated tags)

    def add_documents(self, docs, embs, metas=None):
        """
        Add document chunks, their embeddings, and optional metadata to store.
        docs: list of str
        embs: list of list/np.ndarray
        metas: list of dict (optional)
        """
        if metas is None:
            metas = [{} for _ in range(len(docs))]
            
        for doc, emb, meta in zip(docs, embs, metas):
            self.documents.append(doc)
            # Normalize embedding immediately for faster cosine search via dot product
            emb_arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_arr)
            self.vectors.append(emb_arr / (norm + 1e-9))
            self.metadata.append(meta)

    def search(self, query_emb, k=3, metadata_filter=None):
        """
        Search for top-k nearest neighbors using Cosine Similarity.
        query_emb: list/np.ndarray of query vector
        metadata_filter: dict of key-value matches (e.g. {"category": "policy"})
        """
        if not self.vectors:
            return []
            
        # 1. Normalize query embedding
        q_arr = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        q_normalized = q_arr / (q_norm + 1e-9)

        candidate_indices = []
        
        # 2. Apply metadata filter (Pre-filtering)
        for idx, meta in enumerate(self.metadata):
            if metadata_filter:
                match = True
                for key, val in metadata_filter.items():
                    if meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue  # skip this vector if it doesn't match filter criteria
            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # 3. Compute cosine similarities (since vectors are normalized, dot product = cosine similarity)
        candidate_vectors = np.array([self.vectors[i] for i in candidate_indices])
        scores = np.dot(candidate_vectors, q_normalized)

        # 4. Sort scores in descending order
        sorted_candidates_idx = np.argsort(scores)[::-1][:k]
        
        # 5. Compile results
        results = []
        for idx_in_candidates in sorted_candidates_idx:
            actual_idx = candidate_indices[idx_in_candidates]
            results.append({
                "document": self.documents[actual_idx],
                "score": float(scores[idx_in_candidates]),
                "metadata": self.metadata[actual_idx]
            })
            
        return results

    def save_index(self, filepath):
        """Save vector database index to files."""
        # Save vectors as .npy binary, text and metadata as JSON
        np.save(filepath + "_vectors.npy", np.array(self.vectors))
        with open(filepath + "_meta.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)
        print(f"Index successfully saved to {filepath}_vectors.npy and {filepath}_meta.json")

    def load_index(self, filepath):
        """Load database index from files."""
        self.vectors = list(np.load(filepath + "_vectors.npy"))
        with open(filepath + "_meta.json", "r") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        print("Index loaded successfully.")


def main():
    # Setup Vector Store
    vdb = ScratchVectorStore()

    # Create dummy embeddings (dimension=4) for 4 documents
    docs = [
        "Refunds are processed within 5 business days to the original card.",
        "Exchange items must be sent back in original packaging.",
        "We offer standard shipping (3-5 days) and express shipping (1-2 days).",
        "Express shipping costs 15 dollars, standard shipping is free for orders over 50."
    ]
    
    # Let's generate semantic-like vectors where:
    # Dim 0: Refund topic, Dim 1: Return topic, Dim 2: Shipping speed, Dim 3: Shipping cost
    embs = [
        [1.0, 0.0, 0.0, 0.0],  # Refund policy
        [0.0, 1.0, 0.0, 0.0],  # Return/exchange
        [0.0, 0.0, 1.0, 0.0],  # Shipping speed
        [0.0, 0.0, 0.2, 0.8]   # Shipping costs
    ]
    
    metas = [
        {"category": "billing"},
        {"category": "returns"},
        {"category": "shipping"},
        {"category": "shipping"}
    ]

    vdb.add_documents(docs, embs, metas)

    # 1. Search Query: "How much does fast delivery cost?"
    # High similarity to Dim 2 (shipping speed) and Dim 3 (shipping cost)
    query_emb = [0.0, 0.0, 0.3, 0.7]
    print("--- Searching for: 'How much does fast delivery cost?' ---")
    results = vdb.search(query_emb, k=2)
    for r in results:
        print(f"Score: {r['score']:.4f} | Category: {r['metadata']['category']} | Text: {r['document']}")

    # 2. Search Query with Metadata Filter: shipping category only
    print("\n--- Searching with category='shipping' filter ---")
    results_filtered = vdb.search(query_emb, k=2, metadata_filter={"category": "shipping"})
    for r in results_filtered:
        print(f"Score: {r['score']:.4f} | Category: {r['metadata']['category']} | Text: {r['document']}")

    # 3. Test saving and loading
    print("\n--- Testing serialization ---")
    vdb.save_index("Complete_Course/RAG/implementations/my_scratch_index")
    
    new_vdb = ScratchVectorStore()
    new_vdb.load_index("Complete_Course/RAG/implementations/my_scratch_index")
    
    print("\nSearch results using loaded index:")
    loaded_results = new_vdb.search(query_emb, k=1)
    print(f"Doc: {loaded_results[0]['document']}")

if __name__ == '__main__':
    main()
