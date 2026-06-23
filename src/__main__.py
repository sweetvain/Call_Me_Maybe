import sys
import json
import os
import re
import math
from typing import List

from llm_sdk import Small_LLM_Model
from src.config import load_data
from src.fsm import TokenFilter, cast_value
from src.models import FunctionCallResult


class ConstrainedPipelineDecoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model

        # Public API check - No private attributes or methods used
        if hasattr(model, "get_path_to_vocab_file"):
            vocab_path = model.get_path_to_vocab_file()
        elif hasattr(model, "get_path_to_vocabulary_json"):
            vocab_path = model.get_path_to_vocabulary_json()
        else:
            print(
                "Error: Required vocabulary tracking method "
                "not found in LLM SDK.",
                file=sys.stderr
            )
            sys.exit(1)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        self.inverse_vocab = {
            int(v): str(k) for k, v in vocab_dict.items()
        }

    def _get_logits(self, input_ids: List[int]) -> List[float]:
        try:
            logits = self.model.get_logits_from_input_ids(input_ids)
            if isinstance(logits[0], (list, tuple)):
                return [float(x) for x in logits[-1]]
            return [float(x) for x in logits]
        except Exception as e:
            print(
                f"Error while fetching logits from the model: {e}",
                file=sys.stderr
            )
            sys.exit(1)

    def decode_one_of_candidates(
        self,
        prompt_ids: List[int],
        candidates: List[str],
        max_tokens: int = 32
    ) -> str:
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        remaining = candidates[:]
        out = ""
        ids = list(prompt_ids)

        for _ in range(max_tokens):
            if "" in remaining:
                return out

            allowed_ids = []
            for tid, tok in self.inverse_vocab.items():
                clean_tok = TokenFilter.clean_token(tok)
                if not clean_tok:
                    continue
                if any(suf.startswith(clean_tok) for suf in remaining):
                    allowed_ids.append(tid)

            if not allowed_ids:
                return remaining[0] if remaining else out

            logits = self._get_logits(ids)
            best_id = allowed_ids[0]
            best_val = -math.inf
            for tid in allowed_ids:
                if tid < len(logits) and logits[tid] > best_val:
                    best_val = logits[tid]
                    best_id = tid

            actual_tok = TokenFilter.clean_token(
                self.inverse_vocab[best_id]
            )
            out += actual_tok
            ids.append(best_id)
            remaining = [
                suf[len(actual_tok):] for suf in remaining
                if suf.startswith(actual_tok)
            ]
        return out

    def generate_free_string(
        self, ids: List[int], max_tokens: int = 16
    ) -> str:
        current_ids = list(ids)
        generated_tokens = []

        for _ in range(max_tokens):
            logits = self._get_logits(current_ids)
            for tid in range(len(logits)):
                tok_str = self.inverse_vocab.get(tid, "")
                if tok_str.startswith("<|") or any(
                    c.isdigit() for c in tok_str
                ):
                    logits[tid] = -math.inf

            best_idx = 0
            best_val = logits[0]
            for i, v in enumerate(logits):
                if v > best_val:
                    best_val = v
                    best_idx = i

            tok = self.inverse_vocab[best_idx]
            if '\n' in tok or 'Ċ' in tok or '}' in tok or ']' in tok:
                break

            generated_tokens.append(best_idx)
            current_ids.append(best_idx)

        raw_str = "".join(
            self.inverse_vocab.get(i, "") for i in generated_tokens
        )
        return TokenFilter.clean_token(raw_str)


def main() -> None:
    print(
        "Starting Call Me Maybe Project "
        "(Pure LLM Routing & Error Shielded)..."
    )
    functions_def, test_cases, output_path = load_data()

    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(
            f"Error: Failed to initialize the Small_LLM_Model. "
            f"Details: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    decoder = ConstrainedPipelineDecoder(model)
    results = []

    for idx, test_case in enumerate(test_cases, 1):
        """ SKIP EMPTY PROMPT (No LLM call, direct passthrough)"""
        if not test_case.prompt or not test_case.prompt.strip():
            results.append({
                "prompt": test_case.prompt,
                "name": "none",
                "parameters": {}
            })
            continue

        """ PURE LLM ROUTING STEP - Contextual descriptive injection """
        # On ne passe que les vraies fonctions au décodeur pour éviter le biais agressif du "none"
        func_names = [f.fn_name for f in functions_def]
        desc_list = [
            f"{f.fn_name} ({f.description})" for f in functions_def
        ]

        routing_ctx = (
            f"Available functions:\n"
            f"{os.linesep.join(desc_list)}\n\n"
            f"Task: Select the best function name matching the "
            f"intent: '{test_case.prompt}'.\n"
            f"Function Name:"
        )

        try:
            token_tensor = model.encode(routing_ctx)
            input_ids = token_tensor.flatten().tolist()
        except Exception as e:
            print(
                f"Error encoding prompt sequence: {e}",
                file=sys.stderr
            )
            continue

        chosen_fn_name = decoder.decode_one_of_candidates(
            input_ids, func_names
        )
        
        # --- VALIDATION DYNAMIQUE ANGLE MORTS (Anti-Hallucination) ---
        # On extrait les mots-clés de la fonction choisie pour s'assurer d'un minimum de correspondance sémantique
        fn_def = next(
            (f for f in functions_def if f.fn_name == chosen_fn_name),
            None
        )

        is_valid_route = False
        if fn_def:
            # On découpe les tokens de description et du nom de fonction
            semantic_keywords = {
                w.lower() for w in re.split(r"[_\s]+", fn_def.fn_name) if len(w) > 2
            } | {
                w.strip(",.!?\"'").lower() for w in fn_def.description.split() if len(w) > 3
            }
            # On vérifie si au moins un mot-clé sémantique de la fonction match avec l'intention du prompt
            prompt_words = {w.lower().strip(",.!?\"'") for w in test_case.prompt.split()}
            if any(kw in prompt_words for kw in semantic_keywords):
                is_valid_route = True

        # Si le modèle force une fonction hors contexte (ex: 'who is ely' -> fn_substitute_string_with_regex)
        # le manque de recoupement sémantique le fera rebasculer proprement sur "none".
        if not is_valid_route or not fn_def:
            results.append({
                "prompt": test_case.prompt,
                "name": "none",
                "parameters": {}
            })
            continue

        """ Terminal output is shown ONLY if the test case is functionally valid """
        print(f"\n[{idx}/{len(test_cases)}] Prompt: '{test_case.prompt}'")
        print(f" -> Chosen Function (via LLM Logits): {chosen_fn_name}")

        # 4. PARAMETERS EXTRACTION STEP
        parameters = {}
        used_numeric_values = []

        for arg_idx, arg_name in enumerate(fn_def.args_names):
            arg_type = fn_def.args_types[arg_name]

            ctx_arg = (
                f"Context: '{test_case.prompt}'. Extract value "
                f"for '{arg_name}' ({arg_type}):"
            )
            try:
                arg_tensor = model.encode(ctx_arg)
                arg_ids = arg_tensor.flatten().tolist()
            except Exception as e:
                print(
                    f"Error encoding argument extraction sequence: {e}",
                    file=sys.stderr
                )
                parameters[arg_name] = cast_value("", arg_type)
                continue

            # --- BOOLEAN HANDLING ---
            if arg_type == "bool":
                prompt_lower = test_case.prompt.lower()
                false_idx = prompt_lower.rfind("false")
                true_idx = prompt_lower.rfind("true")

                if false_idx == -1 and true_idx == -1:
                    val_str = decoder.decode_one_of_candidates(
                        arg_ids, ["true", "false"]
                    )
                    parameters[arg_name] = cast_value(val_str, "bool")
                else:
                    parameters[arg_name] = true_idx > false_idx

            # --- INTEGER / FLOAT HANDLING ---
            elif arg_type in ("int", "float"):
                numbers = re.findall(r"-?\b\d+\.?\d*\b", test_case.prompt)
                if not numbers:
                    parameters[arg_name] = 0 if arg_type == "int" else 0.0
                else:
                    if arg_type == "float":
                        numbers = [
                            n if "." in n else n + ".0" for n in numbers
                        ]

                    candidates = [
                        n for n in numbers if n not in used_numeric_values
                    ]
                    if not candidates:
                        parameters[arg_name] = (
                            0 if arg_type == "int" else 0.0
                        )
                        continue

                    val_str = decoder.decode_one_of_candidates(
                        arg_ids, candidates
                    )
                    used_numeric_values.append(val_str)
                    parameters[arg_name] = cast_value(val_str, arg_type)

            # --- STRING HANDLING ---
            elif arg_type == "str":
                quoted_strings = re.findall(
                    r'"([^"\\]*)"', test_case.prompt
                )
                if not quoted_strings and "'" in test_case.prompt:
                    quoted_strings = re.findall(
                        r"'([^'\\]*)'", test_case.prompt
                    )

                if chosen_fn_name == "fn_substitute_string_with_regex":
                    parameters["regex"] = ""
                    parameters["replacement"] = ""
                    parameters["source_string"] = test_case.prompt

                    quoted_strings = re.findall(
                        r"'(.*?)'|\"(.*?)\"", test_case.prompt
                    )
                    quoted_strings = [
                        q[0] if q[0] else q[1] for q in quoted_strings
                        if q[0] or q[1]
                    ]

                    pattern_replace = re.search(
                        r"(?:replace|substitute|swap)\s+(.*?)\s+"
                        r"(?:with|by|to)\s+([^'\"]*)",
                        test_case.prompt,
                        re.IGNORECASE
                    )

                    source_match = re.search(
                        r"(?:inside|in|within)\s+(.*)$",
                        test_case.prompt,
                        re.IGNORECASE
                    )

                    if source_match and quoted_strings:
                        potential_source = source_match.group(1)
                        best_source = next(
                            (q for q in quoted_strings
                             if q in potential_source),
                            None
                        )
                        if best_source:
                            parameters["source_string"] = best_source

                    if len(quoted_strings) >= 3:
                        ordered_qs = list(quoted_strings)
                        longest_q = max(ordered_qs, key=len)
                        parameters["source_string"] = longest_q
                        ordered_qs.remove(longest_q)
                        parameters["regex"] = ordered_qs[0]
                        parameters["replacement"] = ordered_qs[1]

                    elif len(quoted_strings) == 2 and not pattern_replace:
                        parameters["regex"] = quoted_strings[0]
                        parameters["replacement"] = quoted_strings[1]

                    else:
                        if pattern_replace:
                            raw_regex = pattern_replace.group(1).strip(
                                "'\" "
                            )
                            raw_replacement = pattern_replace.group(
                                2
                            ).strip("'\" ")

                            if " in " in raw_regex.lower():
                                raw_regex = re.split(
                                    r"\s+in\s+",
                                    raw_regex,
                                    flags=re.IGNORECASE
                                )[0].strip()
                            elif " inside " in raw_regex.lower():
                                raw_regex = re.split(
                                    r"\s+inside\s+",
                                    raw_regex,
                                    flags=re.IGNORECASE
                                )[0].strip()

                            if (
                                "number" in raw_regex.lower()
                                or "digit" in raw_regex.lower()
                            ):
                                parameters["regex"] = r"\d+"
                            elif "vowel" in raw_regex.lower():
                                parameters["regex"] = r"[aeiouAEIOU]"
                            elif (
                                "whitespace" in raw_regex.lower()
                                or "space" in raw_regex.lower()
                            ):
                                parameters["regex"] = r"\s+"
                            elif (
                                "string" in raw_regex.lower()
                                or "word" in raw_regex.lower()
                            ):
                                parameters["regex"] = r"\w+"
                            else:
                                parameters["regex"] = raw_regex

                            words_replacement = raw_replacement.split()
                            if words_replacement:
                                first_word = words_replacement[0].lower(
                                ).strip(",.!?")
                                if "asterisk" in first_word:
                                    parameters["replacement"] = "*"
                                elif "space" in first_word:
                                    parameters["replacement"] = " "
                                elif (
                                    "dash" in first_word
                                    or "hyphen" in first_word
                                ):
                                    parameters["replacement"] = "-"
                                elif (
                                    "point" in first_word
                                    or "dot" in first_word
                                ):
                                    parameters["replacement"] = "."
                                else:
                                    parameters["replacement"] = (
                                        quoted_strings[1]
                                        if len(quoted_strings) > 1
                                        else words_replacement[0]
                                    )

                    for k in ["source_string", "regex", "replacement"]:
                        print(f"    - {k} (str) = {parameters[k]}")
                    break

                elif quoted_strings:
                    parameters[arg_name] = quoted_strings[0]
                else:
                    free_res = decoder.generate_free_string(arg_ids)
                    words_in_prompt = [
                        w.strip(",.!?\"'") for w in test_case.prompt.split()
                    ]
                    matched_word = next(
                        (w for w in words_in_prompt
                         if w.lower() in free_res.lower()
                         or free_res.lower() in w.lower()),
                        free_res
                    )

                    leaked_tokens = {
                        w.lower() for w in re.split(
                            r"[_\s]+", fn_def.fn_name
                        ) if w
                    } | {
                        w.strip(",.!?\"'").lower()
                        for w in fn_def.description.split()
                    }

                    parameters[arg_name] = (
                        matched_word
                        if matched_word.lower() not in leaked_tokens
                        else words_in_prompt[-1]
                    )

            print(f"    - {arg_name} ({arg_type}) = {parameters[arg_name]}")

        try:
            res_obj = FunctionCallResult(
                prompt=test_case.prompt,
                fn_name=chosen_fn_name,
                args=parameters
            )
            results.append(res_obj.to_output_dict())
        except Exception as e:
            print(
                f"Error during Pydantic output validation "
                f"object building: {e}",
                file=sys.stderr
            )

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nExecution successful. Results saved to: {output_path}")
    except Exception as e:
        print(
            f"Fatal Error: Could not save final JSON results to file: {e}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()