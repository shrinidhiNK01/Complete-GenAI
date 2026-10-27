from app.agents.tools import safe_calculate


def test_safe_calculate_addition():
    assert safe_calculate("2 + 3 * 4") == 14.0

