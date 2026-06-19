from enum import Enum, auto
from typing import List, Dict, Any, Optional
from src.models import FunctionDefinition


class JSONState(Enum):
    START = auto()
    PROMPT_KEY = auto()
    PROMPT_VALUE = auto()
    NAME_KEY = auto()
    NAME_VALUE = auto()
    PARAMETERS_KEY = auto()
    PARAM_NAME = auto()
    PARAM_COLON = auto()
    PARAM_VALUE = auto()
    PARAM_COMMA_OR_CLOSE = auto()
    END = auto()


class TokenFilter:
    @staticmethod
    def clean_token(token_text: str) -> str:
        return token_text.replace("Ġ", " ").replace(" ", " ").strip()


def cast_value(value: str, target_type: str) -> Any:
    """Convertit proprement les buffers extraits par la FSM dans le type cible."""
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