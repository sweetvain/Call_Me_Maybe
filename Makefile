PYTHON = uv run python
SRC = src

install:
	uv sync

run:
	$(PYTHON) -m $(SRC) --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache llm_sdk/__pycache__ src/__pycache__

lint:
	flake8 .
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	flake8 .
	mypy --strict .