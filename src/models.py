from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


ArgType = Literal["int", "float", "str", "bool"]


class FunctionDefinition(BaseModel):
    """Represents the definition of an injectable function."""
    fn_name: str
    description: str = "No description provided"
    args_names: List[str]
    args_types: Dict[str, ArgType]
    return_type: ArgType | Literal["void"] | str

    def model_post_init(self, __context: Any) -> None:
        """Validates the exact correspondence between
        argument names and types."""
        if not self.args_names and self.args_types:
            raise ValueError(
                "args_types must be empty if args_names is empty."
            )
        if (
            self.args_names
            and set(self.args_types.keys()) != set(self.args_names)
        ):
            raise ValueError(
                "args_types keys must exactly match args_names."
            )


class TestPrompt(BaseModel):
    """Represents an input test case."""
    prompt: str


class FunctionCallResult(BaseModel):
    """Strict output format required by the subject."""
    prompt: str
    fn_name: str = Field(serialization_alias="name")
    args: Dict[str, Any] = Field(
        default_factory=dict, serialization_alias="parameters"
    )

    def to_output_dict(self) -> Dict[str, Any]:
        """Exports the model respecting the required
        serialization aliases."""
        return self.model_dump(by_alias=True)
