"""
Query a FAISS index and generate an answer using a small text-generation model.

Usage:
  python query_rag.py --index_path ../faiss_index.index --data_path ../faiss_data.pkl --question "What is X?"
"""
import os
import argparse
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

try:
    import faiss
except Exception:
    faiss = None


def load_index(index_path, data_path, embed_model_name='all-MiniLM-L6-v2'):
    if faiss is None:
        raise ImportError('faiss is required (install faiss-cpu).')
    index = faiss.read_index(index_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    model = SentenceTransformer(embed_model_name)
    return index, data, model


def retrieve(index, data, encoder, query, top_k=5):
    q_emb = encoder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    passages = [data['passages'][int(i)] for i in I[0]]
    return passages


def generate_answer(passages, question, gen_model_name='gpt2'):
    context = '\n\n'.join(passages)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    gen = pipeline('text-generation', model=gen_model_name)
    out = gen(prompt, max_length=200, do_sample=True, temperature=0.7)
    return out[0]['generated_text']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    index, data, encoder = load_index(args.index_path, args.data_path)
    passages = retrieve(index, data, encoder, args.question, top_k=args.top_k)
    ans = generate_answer(passages, args.question)
    print(ans)


if __name__ == '__main__':
    main()
