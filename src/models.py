from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

# Types de données primitifs acceptés par l'énoncé
ArgType = Literal["int", "float", "str", "bool"]


class FunctionDefinition(BaseModel):
    """Représente la définition d'une fonction injectable."""
    fn_name: str
    description: str = "No description provided"  # <-- Ajouté ici, simple et propre
    args_names: List[str]
    args_types: Dict[str, ArgType]
    return_type: ArgType | Literal["void"] | str

    def model_post_init(self, __context: Any) -> None:
        """Valide la parfaite correspondance entre les noms et les types d'arguments."""
        if not self.args_names and self.args_types:
            raise ValueError("args_types doit être vide si args_names est vide.")
        if self.args_names and set(self.args_types.keys()) != set(self.args_names):
            raise ValueError("Les clés de args_types doivent correspondre exactement à args_names.")


class TestPrompt(BaseModel):
    """Représente un cas de test d'entrée."""
    prompt: str


class FunctionCallResult(BaseModel):
    """Format de sortie strict exigé par le sujet."""
    prompt: str
    fn_name: str = Field(serialization_alias="name")
    args: Dict[str, Any] = Field(default_factory=dict, serialization_alias="parameters")

    def to_output_dict(self) -> Dict[str, Any]:
        """Exporte le modèle en respectant les alias de sérialisation requis."""
        return self.model_dump(by_alias=True)