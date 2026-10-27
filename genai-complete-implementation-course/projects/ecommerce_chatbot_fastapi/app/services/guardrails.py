RISKY_TERMS = {
    "refund now",
    "issue refund",
    "cancel order",
    "change address",
    "delete account",
    "chargeback",
    "legal",
}


def needs_human_review(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in RISKY_TERMS)


def redact_sensitive(text: str) -> str:
    words = []
    for word in text.split():
        if "@" in word:
            words.append("[redacted-email]")
        elif len(word) >= 12 and any(char.isdigit() for char in word):
            words.append("[redacted-token]")
        else:
            words.append(word)
    return " ".join(words)

