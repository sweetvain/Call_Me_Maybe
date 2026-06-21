import os
import json
import pytest
import sys
from unittest.mock import patch, MagicMock

# Importation des fonctions cibles du projet
from src.config import load_data
from src.fsm import cast_value, TokenFilter

# =========================================================================
# TESTS DES CAS LIMITES DE TRANSTYPAGE (src/fsm.py)
# =========================================================================

def test_cast_value_integer():
    """Test standard and edge cases for casting integers."""
    assert cast_value("42", "int") == 42
    assert cast_value("42.7", "int") == 42  # Truncation verification
    assert cast_value("invalid_str", "int") == 0  # Robustness to malformed input

def test_cast_value_float():
    """Test standard and edge cases for casting floats."""
    assert cast_value("3.14", "float") == 3.14
    assert cast_value("10", "float") == 10.0
    assert cast_value("bad_float", "float") == 0.0  # Robustness check

def test_cast_value_boolean():
    """Test boolean conversions matching schema requirements."""
    assert cast_value("true", "bool") is True
    assert cast_value("1", "bool") is True
    assert cast_value("FALSE", "bool") is False
    assert cast_value("random_text", "bool") is False

def test_token_filter_cleaning():
    """Verify BPE specific whitespace token artifacts cleaning."""
    assert TokenFilter.clean_token("Ġfunction_name") == "function_name"
    assert TokenFilter.clean_token(" normal ") == "normal"


# =========================================================================
# TESTS DE GESTION DES ERREURS D'ENTRÉE (src/config.py)
# =========================================================================

@pytest.fixture
def create_valid_temp_files(tmp_path):
    """Fixture creating valid temp functions and tests structures."""
    func_file = tmp_path / "funcs.json"
    test_file = tmp_path / "tests.json"
    
    funcs_data = [
        {
            "fn_name": "fn_add",
            "fn_description": "Adds two numbers",
            "args_names": ["a", "b"],
            "args_types": {"a": "int", "b": "int"},
            "return_type": "int"
        }
    ]
    tests_data = [{"prompt": "Add 2 and 3"}]
    
    func_file.write_text(json.dumps(funcs_data), encoding="utf-8")
    test_file.write_text(json.dumps(tests_data), encoding="utf-8")
    
    return str(func_file), str(test_file)

@patch("src.config.parse_arguments")
def test_load_data_success(mock_args, create_valid_temp_files):
    """Verifies that flawless data structures parse correctly without crashing."""
    func_path, test_path = create_valid_temp_files
    
    # Simulating command line args inputs
    mock_args.return_value = MagicMock(
        functions_definition=func_path,
        input=test_path,
        output="output/results.json"
    )
    
    functions, prompts, output_path = load_data()
    
    assert len(functions) == 1
    assert functions[0].fn_name == "fn_add"
    assert len(prompts) == 1
    assert prompts[0].prompt == "Add 2 and 3"
    assert output_path == "output/results.json"

@patch("src.config.parse_arguments")
def test_load_data_missing_file(mock_args, tmp_path):
    """Ensures code terminates cleanly with exit code 1 if a file is missing."""
    mock_args.return_value = MagicMock(
        functions_definition=str(tmp_path / "ghost_file_1.json"),
        input=str(tmp_path / "ghost_file_2.json"),
        output="output/out.json"
    )
    
    with pytest.raises(SystemExit) as exc_info:
        load_data()
        
    assert exc_info.value.code == 1

@patch("src.config.parse_arguments")
def test_load_data_malformed_json_syntax(mock_args, tmp_path):
    """Ensures strict tracking and termination upon encountering invalid JSON syntax."""
    bad_func_file = tmp_path / "bad_funcs.json"
    good_test_file = tmp_path / "good_tests.json"
    
    # Injection of broken syntax (missing closing bracket)
    bad_func_file.write_text("[{'fn_name': 'unclosed'", encoding="utf-8")
    good_test_file.write_text("[]", encoding="utf-8")
    
    mock_args.return_value = MagicMock(
        functions_definition=str(bad_func_file),
        input=str(good_test_file),
        output="output/out.json"
    )
    
    with pytest.raises(SystemExit) as exc_info:
        load_data()
        
    assert exc_info.value.code == 1