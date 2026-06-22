from typing import Any


class TokenFilter:
    @staticmethod
    def clean_token(token_text: str) -> str:
        return token_text.replace("Ġ", " ").replace(" ", " ").strip()


def cast_value(value: str, target_type: str) -> Any:
    clean_val = value.strip().strip('"').strip("'")
    if target_type == "int":
        try:
            return int(float(clean_val))
        except ValueError:
            return 0
    elif target_type in ("float", "number"):
        try:
            return float(clean_val)
        except ValueError:
            return 0.0
    elif target_type == "bool":
        return clean_val.lower() in ("true", "1")
    return clean_val
