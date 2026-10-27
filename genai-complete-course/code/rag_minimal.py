from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS = [
    {
        "source": "refund_policy.txt",
        "text": "Refunds are allowed within 30 days of purchase if the customer has proof of purchase.",
    },
    {
        "source": "shipping_policy.txt",
        "text": "Standard shipping usually takes 3 to 5 business days.",
    },
    {
        "source": "support_policy.txt",
        "text": "Premium support is available for enterprise customers 24 hours a day.",
    },
]


def retrieve(question: str, top_k: int = 2):
    texts = [doc["text"] for doc in DOCS]
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]
    ranked_indexes = scores.argsort()[::-1][:top_k]
    return [
        {
            "score": float(scores[index]),
            "source": DOCS[index]["source"],
            "text": DOCS[index]["text"],
        }
        for index in ranked_indexes
    ]


if __name__ == "__main__":
    query = input("Question: ")
    results = retrieve(query)
    for result in results:
        print(f"\nScore: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print(result["text"])

