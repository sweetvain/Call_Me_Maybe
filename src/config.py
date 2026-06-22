import argparse
import json
import os
import sys
from typing import List, Tuple
from src.models import FunctionDefinition, TestPrompt


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with mandatory default values."""
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Configuration Loader"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input tests JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/function_calling_results.json",
        help="Path to the output results JSON file"
    )
    return parser.parse_args()


def load_data() -> Tuple[List[FunctionDefinition], List[TestPrompt], str]:
    """Load, adapt, validate, and distribute project input data
    with strict error handling."""
    args = parse_arguments()

    # 1. Management of missing files
    if not os.path.exists(args.functions_definition):
        print(
            f"Error: The functions definition file "
            f"'{args.functions_definition}' does not exist.",
            file=sys.stderr
        )
        sys.exit(1)

    if not os.path.exists(args.input):
        print(
            f"Error: The input tests file '{args.input}' "
            f"does not exist.",
            file=sys.stderr
        )
        sys.exit(1)

    # 2. Loading and structural/syntax parsing validation for Functions
    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            try:
                raw_functions = json.load(f)
            except json.JSONDecodeError as jde:
                print(
                    f"Error: Invalid JSON syntax in "
                    f"'{args.functions_definition}': {jde}",
                    file=sys.stderr
                )
                sys.exit(1)

            items = (
                raw_functions
                if isinstance(raw_functions, list)
                else [raw_functions]
            )

            adapted_functions = []
            for item in items:
                if "name" in item and "parameters" in item:
                    raw_params = item.get("parameters", {})

                    # Traduction dynamique :
                    # 'number' -> 'float', 'string' -> 'str'
                    args_types = {}
                    for k, v in raw_params.items():
                        t = v.get("type", "str")
                        if t == "number":
                            args_types[k] = "float"
                        elif t == "string":
                            args_types[k] = "str"
                        elif t in ("boolean", "bool"):
                            args_types[k] = "bool"
                        else:
                            args_types[k] = t

                    ret_block = item.get("returns", {})
                    ret_type = ret_block.get("type", "void")
                    if ret_type == "number":
                        ret_type = "float"
                    elif ret_type == "string":
                        ret_type = "str"
                    elif ret_type in ("boolean", "bool"):
                        ret_type = "bool"

                    adapted_item = {
                        "fn_name": item["name"],
                        "description": item.get(
                            "description", "No description provided"
                        ),  # <-- Extrait ici
                        "args_names": list(raw_params.keys()),
                        "args_types": args_types,
                        "return_type": ret_type
                    }
                else:
                    adapted_item = item

                adapted_functions.append(
                    FunctionDefinition(**adapted_item)
                )

    except Exception as e:
        print(
            f"Error: Validation failed for functions "
            f"definition schema: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    # 3. Loading and structural/syntax parsing validation for Tests
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            try:
                data_inputs = json.load(f)
            except json.JSONDecodeError as jde:
                print(
                    f"Error: Invalid JSON syntax in "
                    f"'{args.input}': {jde}",
                    file=sys.stderr
                )
                sys.exit(1)

            items_inputs = (
                data_inputs
                if isinstance(data_inputs, list)
                else [data_inputs]
            )
            prompts = [TestPrompt(**item) for item in items_inputs]

    except Exception as e:
        print(
            f"Error: Validation failed for input tests schema: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    return adapted_functions, prompts, args.output
