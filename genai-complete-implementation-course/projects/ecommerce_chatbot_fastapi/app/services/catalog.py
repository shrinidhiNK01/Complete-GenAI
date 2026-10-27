import json
from pathlib import Path
from typing import Dict, List


class ProductCatalog:
    def __init__(self, data_path: Path):
        self.products = json.loads(data_path.read_text(encoding="utf-8"))

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        terms = set(query.lower().split())
        scored = []
        for product in self.products:
            haystack = " ".join(
                [
                    product["name"],
                    product["category"],
                    product["description"],
                    " ".join(product["tags"]),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, product))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [product for _score, product in scored[:limit]]

    def format_products(self, products: List[Dict]) -> str:
        if not products:
            return "I could not find matching products."
        lines = []
        for product in products:
            lines.append(
                f"{product['name']} ({product['sku']}): {product['description']} Price: {product['price']}."
            )
        return "\n".join(lines)

