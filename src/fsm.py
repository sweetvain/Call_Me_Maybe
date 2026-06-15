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

class JSONStateMachine:
    def __init__(self, target_prompt: str, available_functions: List[FunctionDefinition]):
        self.target_prompt = target_prompt
        self.functions: Dict[str, FunctionDefinition] = {f.fn_name: f for f in available_functions}
        
        self.current_state: JSONState = JSONState.START
        self.generated_text: str = ""
        
        self.chosen_function: Optional[str] = None
        self.current_param_name: Optional[str] = None
        self.processed_params: List[str] = []
        
        # Accumulateurs de valeurs pour piloter les types
        self.param_value_buffer = ""

    def get_allowed_next_strings(self) -> List[str]:
        """
        Renvoie la ou les chaînes attendues par l'automate au stade actuel.
        """
        if self.current_state == JSONState.START:
            return ['{"prompt": "']
            
        if self.current_state == JSONState.PROMPT_KEY:
            return [self.target_prompt + '", "name": "']
            
        if self.current_state == JSONState.PROMPT_VALUE:
            # Transition de secours si gérée token par token
            return ['", "name": "']
            
        if self.current_state == JSONState.NAME_KEY:
            # On autorise uniquement les noms des fonctions disponibles
            return [f'{fn_name}", "parameters": {{' for fn_name in self.functions.keys()]
            
        if self.current_state == JSONState.NAME_VALUE:
            return ['", "parameters": {']
            
        if self.current_state == JSONState.PARAMETERS_KEY:
            if not self.chosen_function:
                return ["}"]
            fn_def = self.functions[self.chosen_function]
            # Si la fonction n'attend aucun paramètre
            if not fn_def.args_names:
                return ["}}"]
            # Sinon, on attend le premier paramètre
            first_param = fn_def.args_names[0]
            return [f'"{first_param}": ']
            
        if self.current_state == JSONState.PARAM_NAME:
            if not self.chosen_function:
                return ["}"]
            fn_def = self.functions[self.chosen_function]
            
            # Trouver le prochain paramètre non traité
            remaining = [p for p in fn_def.args_names if p not in self.processed_params and p != self.current_param_name]
            if remaining:
                return [f'"{remaining[0]}": ']
            return ["}"]

        if self.current_state == JSONState.PARAM_COLON:
            return [": "]

        if self.current_state == JSONState.PARAM_VALUE:
            if not self.chosen_function or not self.current_param_name:
                return []
            
            p_type = self.functions[self.chosen_function].args_types[self.current_param_name]
            
            # Si on commence juste à générer la valeur
            if not self.param_value_buffer:
                if p_type == "str":
                    return ['"']  # Obligation d'ouvrir par un guillemet
                if p_type == "bool":
                    return ["true", "false"]
                if p_type in ("int", "float"):
                    return ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            
            # Si on est déjà en train d'écrire la valeur
            if p_type == "str":
                # Si le string n'est pas refermé, on est en texte libre
                if not (self.param_value_buffer.startswith('"') and self.param_value_buffer.endswith('"') and len(self.param_value_buffer) > 1):
                    return ["ANY_STR_CHAR"]
            
            # Par défaut, si l'analyse par token individuel s'en charge :
            return ["ANY_STR_CHAR"]

        if self.current_state == JSONState.PARAM_COMMA_OR_CLOSE:
            if not self.chosen_function:
                return ["}"]
            fn_def = self.functions[self.chosen_function]
            
            # S'il reste des paramètres à traiter -> virgule attendue
            remaining = [p for p in fn_def.args_names if p not in self.processed_params]
            if remaining:
                return [", "]
            return ["}}"]

        if self.current_state == JSONState.END:
            return []
            
        return []

    def consume(self, text: str) -> None:
        """
        Consomme le texte sélectionné par le LLM et met à jour l'état de la FSM.
        """
        self.generated_text += text

        if self.current_state == JSONState.START:
            if self.generated_text.endswith('{"prompt": "'):
                self.current_state = JSONState.PROMPT_KEY
            return

        if self.current_state == JSONState.PROMPT_KEY:
            if self.generated_text.endswith('", "name": "'):
                self.current_state = JSONState.NAME_KEY
            return

        if self.current_state == JSONState.NAME_KEY:
            if '", "parameters": {' in self.generated_text:
                # Détection de la fonction choisie
                for fn_name in self.functions.keys():
                    if f'"name": "{fn_name}"' in self.generated_text:
                        self.chosen_function = fn_name
                        break
                self.current_state = JSONState.PARAM_NAME
            return

        if self.current_state == JSONState.PARAM_NAME:
            if self.generated_text.endswith('}}'):
                self.current_state = JSONState.END
                return
                
            # Extraction du paramètre en cours d'écriture
            for param in self.functions[self.chosen_function].args_names:
                if self.generated_text.endswith(f'"{param}": '):
                    self.current_param_name = param
                    self.current_state = JSONState.PARAM_VALUE
                    self.param_value_buffer = ""
                    break
            return

        if self.current_state == JSONState.PARAM_VALUE:
            self.param_value_buffer += text
            p_type = self.functions[self.chosen_function].args_types[self.current_param_name]
            
            # Vérification de complétude de la valeur
            is_complete = False
            if p_type == "str":
                if self.param_value_buffer.startswith('"') and self.param_value_buffer.endswith('"') and len(self.param_value_buffer) > 1:
                    is_complete = True
            elif p_type == "bool":
                if self.param_value_buffer in ("true", "false"):
                    is_complete = True
            elif p_type in ("int", "float"):
                # Pour les nombres, on attend que le LLM produise la suite du JSON (virgule ou fin)
                if text in (", ", "}", " }"):
                    is_complete = True
                    # On retire le caractère de structure du buffer de valeur
                    self.param_value_buffer = self.param_value_buffer[:-len(text)]
            
            if is_complete:
                self.processed_params.append(self.current_param_name)
                
                # Ajustement de l'état suivant en fonction du caractère de coupure rencontré
                if text == ", " or self.generated_text.endswith(", "):
                    self.current_param_name = None
                    self.current_state = JSONState.PARAM_NAME
                elif text in ("}", "}}") or self.generated_text.endswith("}}"):
                    self.current_state = JSONState.END
                else:
                    self.current_state = JSONState.PARAM_COMMA_OR_CLOSE
            return

        if self.current_state == JSONState.PARAM_COMMA_OR_CLOSE:
            if self.generated_text.endswith(", "):
                self.current_param_name = None
                self.current_state = JSONState.PARAM_NAME
            elif self.generated_text.endswith("}}") or text == "}":
                self.current_state = JSONState.END
            return