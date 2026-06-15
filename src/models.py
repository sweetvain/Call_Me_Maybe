from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str


class TestPrompt(BaseModel):
    prompt: str


# Représentation du fichier final attendu en sortie
class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]