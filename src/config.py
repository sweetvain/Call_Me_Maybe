import argparse
import json
import os
import sys
from typing import List
from pydantic import ValidationError
from src.models import FunctionDefinition, TestPrompt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Constrained JSON Generation")
    parser.add_argument("--functions_definition", required=True, help="Path to functions definition JSON")
    parser.add_argument("--input", required=True, help="Path to input tests JSON")
    parser.add_argument("--output", required=True, help="Path to save the output JSON")
    return parser.parse_args()


def load_data() -> tuple[List[FunctionDefinition], List[TestPrompt], str]:
    args = parse_arguments()
    
    # Validation du fichier des définitions
    if not os.path.exists(args.functions_definition):
        print(f"Erreur : Le fichier {args.functions_definition} n'existe pas.", file=sys.stderr)
        sys.exit(1)
        
    # Validation du fichier d'input
    if not os.path.exists(args.input):
        print(f"Erreur : Le fichier {args.input} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.functions_definition, 'r', encoding='utf-8') as f:
            defs_data = json.load(f)
        functions = [FunctionDefinition(**item) for item in defs_data]
        
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        prompts = [TestPrompt(**item) for item in input_data]
        
    except json.JSONDecodeError as e:
        print(f"Erreur : Fichier JSON malformé. Détails : {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Erreur : Validation des données échouée. Le schéma ne correspond pas. Détails : {e}", file=sys.stderr)
        sys.exit(1)

    return functions, prompts, args.output