*This project has been created as part of the 42 curriculum by elal-hai.*

# Call Me Maybe — Pure LLM Routing & Constrained Decoding Pipeline

## Description

**Call Me Maybe** is a software engineering project whose goal is to build a fully local **Function Calling** system using a lightweight language model (`Qwen3-0.6B`). The program analyzes natural language requests, automatically selects the most appropriate function from a predefined set of available functions, and extracts the required arguments for execution.

The main objective is to guarantee reliable, deterministic, and properly structured outputs while relying on **constrained decoding** techniques instead of traditional free-text generation.

### Features

* Automatic function selection from natural language prompts.
* Typed argument extraction.
* Logit-based constrained decoding.
* Strict data validation using Pydantic.
* Out-of-scope request detection through semantic validation.
* Fully local pipeline with no dependency on external APIs.

---

## Instructions

### Requirements

* Python 3.11+
* uv
* torch
* transformers
* pydantic

### Installation

```bash
git clone <repository_url>
cd call-me-maybe
uv sync
```

### Execution

To run the evaluation pipeline and generate the output JSON file:

```bash
DEBUG_ROUTING=1 uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Project Structure

```text
data/
├── input/
│   ├── functions_definition.json
│   └── function_calling_tests.json
└── output/
    └── function_calling_results.json

src/
├── __init__.py
├── __main_.py
├── config.py
├── fsm.py
└── models.py
```

---

## Algorithm Explanation

The core of the project is based on a **constrained decoding** approach that provides strict control over the model's outputs.

### 1. Function Routing Through Constrained Decoding

When selecting which function to call, the model is not allowed to generate arbitrary text.

At each generation step:

1. The raw logits are retrieved from the model.
2. A mask is applied to all invalid tokens.
3. Only tokens matching a valid prefix of the remaining candidate function names are allowed.
4. The model is therefore mathematically constrained to generate only existing function names.

This effectively transforms function selection into a finite state machine (FSM) guided decoding process.

### 2. Argument Extraction

Once a function has been selected:

* The model extracts the required arguments.
* Argument types are checked.
* Values are validated using Pydantic.

### 3. Semantic Validation

To prevent false positives when a prompt is unrelated to the available functions:

1. Keywords are extracted from the selected function's name and description.
2. Those keywords are compared against the user's prompt.
3. If no meaningful semantic overlap is detected, the function is rejected.
4. The system falls back to `"none"`.

This additional validation significantly improves robustness on unrelated requests.

---

## Design Decisions

### Removing "none" From Initial Routing

During early testing, the model frequently selected `"none"` whenever it lacked confidence.

To address this issue:

* only actual functions are included during constrained decoding;
* the decision to return `"none"` is deferred until semantic validation.

This approach greatly reduces the bias observed with small language models.

### Strict Validation With Pydantic

All outputs are validated before being accepted.

Benefits include:

* type consistency;
* early error detection;
* guaranteed JSON structure.

### Fully Local Pipeline

The project was intentionally designed to run entirely locally in order to provide:

* reproducibility;
* privacy;
* full control over the inference pipeline.

---

## Performance Analysis

### Accuracy

Testing demonstrated strong performance on:

* standard function calls;
* numerical arguments;
* string manipulation tasks;
* unknown names;
* unrelated prompts.

### Speed

Logit masking drastically reduces the model's search space.

As a result:

* fewer candidate tokens are evaluated;
* inference is faster;
* output stability improves.

### Reliability

The combination of:

* constrained decoding;
* Pydantic validation;
* semantic verification;

provides highly reliable and consistently structured outputs.

---

## Challenges Faced

### Bias Toward "none"

One of the main challenges was the model's tendency to select `"none"` whenever it encountered unfamiliar entities.

Example:

```text
Greet shrek
```

Since "shrek" may not carry enough semantic weight for a small model, the request could incorrectly be classified as unrelated.

### Solution

The issue was solved by:

* removing `"none"` from the constrained routing candidates;
* introducing a separate semantic validation step.

Together, these changes significantly improved routing accuracy.

### Controlling Generation

Small language models occasionally generate:

* unexpected characters;
* special tokens;
* premature end-of-sequence markers.

Dynamic logit filtering was implemented to prevent these outputs from corrupting the decoding process.

---

## Testing Strategy

The implementation was validated using several categories of tests.

### Functional Tests

* arithmetic operations;
* string transformations;
* regular expression replacements;
* multi-argument functions.

### Robustness Tests

* unknown names;
* prompt variations;
* alternative phrasings.

### Out-of-Scope Tests

Examples:

```text
Who is Elon Musk?
What is the weather today?
Tell me a joke
```

The expected behavior is to return `"none"`.

### Automated Validation

Generated results are compared against expected outputs to verify:

* correct function selection;
* valid arguments;
* proper JSON formatting.

---

## Example Usage

### Example 1

Input:

```text
What is the sum of 265 and 345?
```

Output:

```text
Function: fn_add_numbers

a = 265
b = 345
```

### Example 2

Input:

```text
Greet shrek
```

Output:

```text
Function: fn_greet

name = shrek
```

### Example 3

Input:

```text
Replace all vowels in 'Programming is fun' with dots
```

Output:

```text
Function: fn_substitute_string_with_regex

source_string = Programming is fun
regex = [aeiouAEIOU]
replacement = .
```

---

## Resources

### Documentation

* Hugging Face Transformers Documentation
* Pydantic v2 Documentation
* Python Official Documentation
* PyTorch Documentation

### Articles and References

* Constrained Decoding documentation and research papers.
* Finite State Machine (FSM) references.
* Structured output and Function Calling resources.

---

## Use of AI

Artificial Intelligence was used as a technical assistant during the development of this project.

AI assistance was used for:

* code formatting and compliance with coding standards;
* code cleanup and refactoring;
* brainstorming and validating algorithmic ideas;
* discussing constrained decoding strategies;
* reviewing and improving documentation.

All implementation decisions, development work, testing, debugging, and final validation were performed and verified by the project's author.
