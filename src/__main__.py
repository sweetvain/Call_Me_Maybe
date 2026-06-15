import sys
import json
import os
import numpy as np
from typing import List, Dict, Any

from llm_sdk import Small_LLM_Model
from src.config import load_data
from src.fsm import JSONStateMachine, JSONState

def mask_logits(logits: List[float], allowed_strings: List[str], inverse_vocab: Dict[int, str]) -> List[float]:
    """
    Filtre les logits avec des conditions simples et directes.
    """
    filtered_logits = list(logits)
    
    # Si l'automate attend du texte libre, on laisse le LLM générer ce qu'il veut
    if "ANY_STR_CHAR" in allowed_strings:
        return filtered_logits

    for token_id in range(len(filtered_logits)):
        token_text = inverse_vocab.get(token_id, "")
        clean_text = token_text.replace("Ġ", " ") # Nettoyage de l'espace BPE
        
        # Condition simple : on élimine les espaces vides ou les jetons techniques
        if not clean_text.strip() or token_text.startswith("<|"):
            filtered_logits[token_id] = -float("inf")
            continue
            
        # Condition de validation : le jeton doit matcher le début d'une des chaînes autorisées
        valid = False
        for allowed in allowed_strings:
            if allowed.startswith(clean_text) or clean_text.startswith(allowed):
                valid = True
                break
                
        if not valid:
            filtered_logits[token_id] = -float("inf")
            
    return filtered_logits

def main():
    print("Démarrage du projet Call Me Maybe...")
    
    functions, test_prompts, output_path = load_data()
    
    print("Chargement du modèle LLM (Qwen)...")
    model = Small_LLM_Model()
    
    tokenizer = model._tokenizer
    inverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    
    results = []

    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Traitement du prompt : '{test_case.prompt}'")
        
        fsm = JSONStateMachine(target_prompt=test_case.prompt, available_functions=functions)
        input_ids = tokenizer.encode(test_case.prompt)
        
        tokens_generated = 0
        max_new_tokens = 256
        
        while fsm.current_state != JSONState.END and tokens_generated < max_new_tokens:
            raw_logits = model.get_logits_from_input_ids(input_ids)
            allowed_strings = fsm.get_allowed_next_strings()
            
            filtered_logits = mask_logits(raw_logits, allowed_strings, inverse_vocab)
            next_token_id = int(np.argmax(filtered_logits))
            
            # Secours dynamique par injection directe du premier caractère attendu si blocage
            if filtered_logits[next_token_id] == -float("inf"):
                fallback_found = False
                if allowed_strings:
                    target = allowed_strings[0]
                    # On cherche le token le plus court/précis qui valide le début
                    for t_id, t_text in inverse_vocab.items():
                        c_t = t_text.replace("Ġ", " ")
                        if c_t and target.startswith(c_t):
                            next_token_id = t_id
                            fallback_found = True
                            break
                if not fallback_found:
                    print("⚠️ Erreur : Blocage absolu de la FSM.", file=sys.stderr)
                    break
            
            token_text = inverse_vocab[next_token_id]
            clean_text = token_text.replace("Ġ", " ")
            
            fsm.consume(clean_text)
            input_ids.append(next_token_id)
            tokens_generated += 1
            
        print(f"-> JSON Généré : {fsm.generated_text}")
        
        try:
            parsed_json = json.loads(fsm.generated_text)
            results.append(parsed_json)
        except json.JSONDecodeError:
            print("⚠️ Attention : Le JSON généré comporte une anomalie structurelle.", file=sys.stderr)
            results.append({"prompt": test_case.prompt, "name": "error", "parameters": {}})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n Traitement terminé avec succès ! Fichier sauvegardé sous : {output_path}")

if __name__ == "__main__":
    main()