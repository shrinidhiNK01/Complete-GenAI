from pathlib import Path

from app.models import ChatRequest, ChatResponse
from app.services.catalog import ProductCatalog
from app.services.guardrails import needs_human_review, redact_sensitive
from app.services.orders import OrderStore
from app.services.policies import PolicyKnowledgeBase


class EcommerceAssistant:
    def __init__(self, data_dir: Path):
        self.catalog = ProductCatalog(data_dir / "products.json")
        self.orders = OrderStore(data_dir / "orders.json")
        self.policies = PolicyKnowledgeBase(data_dir / "policies")

    def route(self, message: str) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ["order", "tracking", "delivered", "shipment"]):
            return "order_lookup"
        if any(word in lowered for word in ["refund", "return", "warranty", "shipping", "policy"]):
            return "policy_rag"
        if any(word in lowered for word in ["buy", "product", "shoe", "bag", "headphone", "recommend"]):
            return "product_search"
        return "fallback"

    def chat(self, request: ChatRequest) -> ChatResponse:
        clean_message = redact_sensitive(request.message)
        route = self.route(clean_message)
        review = needs_human_review(clean_message)

        if route == "order_lookup":
            order_id = self.orders.find_order_id(clean_message)
            if not order_id:
                return ChatResponse(
                    answer="Please provide your order ID so I can check the status.",
                    route=route,
                    sources=["orders.json"],
                    needs_human_review=review,
                )
            order = self.orders.get_order_for_user(order_id, request.user_id)
            if not order:
                return ChatResponse(
                    answer="I could not find that order for your user account.",
                    route=route,
                    sources=["orders.json"],
                    needs_human_review=True,
                )
            return ChatResponse(
                answer=self.orders.format_order(order),
                route=route,
                sources=["orders.json"],
                needs_human_review=review,
            )

        if route == "policy_rag":
            answer, sources = self.policies.answer(clean_message)
            return ChatResponse(answer=answer, route=route, sources=sources, needs_human_review=review)

        if route == "product_search":
            products = self.catalog.search(clean_message)
            return ChatResponse(
                answer=self.catalog.format_products(products),
                route=route,
                sources=["products.json"],
                needs_human_review=review,
            )

        return ChatResponse(
            answer="I can help with products, policies, refunds, shipping, and order tracking.",
            route=route,
            sources=[],
            needs_human_review=review,
        )

