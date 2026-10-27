from pathlib import Path

from app.models import ChatRequest
from app.services.router import EcommerceAssistant


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_product_search_route():
    assistant = EcommerceAssistant(DATA_DIR)
    response = assistant.chat(ChatRequest(user_id="user_1", message="recommend running shoe"))
    assert response.route == "product_search"
    assert "AeroRun" in response.answer


def test_policy_route():
    assistant = EcommerceAssistant(DATA_DIR)
    response = assistant.chat(ChatRequest(user_id="user_1", message="what is the refund policy"))
    assert response.route == "policy_rag"
    assert "30 days" in response.answer


def test_order_authorization_blocks_other_user_order():
    assistant = EcommerceAssistant(DATA_DIR)
    response = assistant.chat(ChatRequest(user_id="user_1", message="track order ORD1002"))
    assert response.route == "order_lookup"
    assert response.needs_human_review is True
    assert "could not find" in response.answer.lower()

