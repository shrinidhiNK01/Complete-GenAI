import json
from pathlib import Path
from typing import Dict, Optional


class OrderStore:
    def __init__(self, data_path: Path):
        orders = json.loads(data_path.read_text(encoding="utf-8"))
        self.orders = {order["order_id"]: order for order in orders}

    def find_order_id(self, message: str) -> Optional[str]:
        for token in message.replace("#", " ").split():
            normalized = token.strip().upper()
            if normalized in self.orders:
                return normalized
        return None

    def get_order_for_user(self, order_id: str, user_id: str) -> Optional[Dict]:
        order = self.orders.get(order_id)
        if not order:
            return None
        if order["user_id"] != user_id:
            return None
        return order

    def format_order(self, order: Dict) -> str:
        return (
            f"Order {order['order_id']} is {order['status']}. "
            f"Carrier: {order['carrier']}. Tracking: {order['tracking_number']}."
        )

