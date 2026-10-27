"""
Build a FAISS index from text files in a folder using SentenceTransformers.

Usage:
  python build_faiss.py --docs_dir ../docs --index_path ../faiss_index.index

Saves:
- FAISS index file (binary)
- Metadata pickle with id->text mapping
"""
import os
import argparse
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


def load_texts_from_folder(docs_dir):
    texts = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith(('.md', '.txt')):
                path = os.path.join(root, fn)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                    texts.append((path, txt))
    return texts


def build_index(docs_dir, index_path, data_path, embed_model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(embed_model_name)
    docs = load_texts_from_folder(docs_dir)
    passages = []
    sources = []
    for path, txt in docs:
        for p in chunk_text(txt):
            passages.append(p)
            sources.append(path)

    if len(passages) == 0:
        raise ValueError('No passages found in docs_dir')

    embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    if faiss is None:
        raise ImportError('faiss is required (install faiss-cpu).')

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # save index and metadata
    faiss.write_index(index, index_path)
    data = {'passages': passages, 'sources': sources}
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    print(f'Index saved to {index_path}')
    print(f'Metadata saved to {data_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', required=True)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--embed_model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    build_index(args.docs_dir, args.index_path, args.data_path, args.embed_model)
