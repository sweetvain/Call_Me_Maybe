import argparse
import json
import os
import sys
from typing import List, Tuple
from src.models import FunctionDefinition, TestPrompt


def parse_arguments() -> argparse.Namespace:
    """Analyse les arguments requis par la ligne de commande."""
    parser = argparse.ArgumentParser(description="Call Me Maybe - Configuration Loader")
    parser.add_argument("--functions_definition", required=True, help="Chemin vers le JSON des fonctions")
    parser.add_argument("--input", required=True, help="Chemin vers le JSON des tests")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour le résultat JSON")
    return parser.parse_args()


def load_data() -> Tuple[List[FunctionDefinition], List[TestPrompt], str]:
    """Charge, valide et distribue les données d'entrée du projet."""
    args = parse_arguments()

    if not os.path.exists(args.functions_definition):
        print(f"Erreur : Le fichier {args.functions_definition} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Erreur : Le fichier {args.input} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            functions = [FunctionDefinition(**item) for item in json.load(f)]

        with open(args.input, "r", encoding="utf-8") as f:
            prompts = [TestPrompt(**item) for item in json.load(f)]

    except Exception as e:
        print(f"Erreur lors du chargement ou de la validation des données : {e}", file=sys.stderr)
        sys.exit(1)

    return functions, prompts, args.output