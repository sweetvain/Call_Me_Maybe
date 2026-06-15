import json
from llm_sdk import Small_LLM_Model


def explorer_vocabulaire():
    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    vocab_path = model.get_path_to_vocabulary_json()
    print(f"Fichier de vocabulaire localise ici: {vocab_path}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print(f"Taille totale du vocabulaire du modele : {len(vocab)} tokens.")

    inverse_vocab = {v: k for k, v in vocab.items()}
    
    # 5. Petite vérification de sécurité : cherchons les tokens critiques
    tokens_cibles = ["{", "}", '"', ":", ",", "fn_add_numbers"]
    print("\n--- Analyse des tokens critiques ---")
    for t in tokens_cibles:
        token_id = vocab.get(t)
        print(f"Caractère exact '{t}' -> Token ID : {token_id}")
        
    print("\n--- 1. Analyse des espaces et préfixes ---")
    # Observons si l'espace est un caractère spécial (ex: 'Ġ' ou ' ')
    # Regardons comment est encodé un guillemet suivi d'un mot
    mots_tests = ['prompt', 'name', 'parameters']
    for mot in mots_tests:
        # Cherchons si le mot existe tel quel
        if mot in vocab:
            print(f"'{mot}' direct -> ID: {vocab[mot]}")
        
        # Cherchons si le mot existe avec un espace devant (on cherche des motifs fréquents)
        for k, v in vocab.items():
            if k.endswith(mot) and len(k) > len(mot):
                print(f"Variante trouvée pour '{mot}': '{k}' -> ID: {v}")

    print("\n--- 2. Comment le modèle va devoir écrire vos fonctions ---")
    fonctions = ["fn_add_numbers", "fn_get_square_root", "fn_greet"]
    for fn in fonctions:
        print(f"\nDécomposition de '{fn}':")
        # Puisque le mot entier n'existe pas (None), on simule une découpe gloutonne grossière :
        caractères_accumulés = ""
        tokens_trouvés = []
        
        # Astuce temporaire pour voir quels morceaux existent dans le vocabulaire :
        for char in fn:
            caractères_accumulés += char
            if caractères_accumulés in vocab:
                # C'est un morceau valide
                pass
        
        # Pour être plus précis, regardons juste si des sous-parties cruciales existent :
        morceaux = ["fn", "_add", "_numbers", "_get", "_square", "_root", "_greet"]
        for m in morceaux:
            if m in vocab:
                print(f"  Sous-morceau '{m}' existe -> ID: {vocab[m]}")
            elif f"Ġ{m}" in vocab:
                print(f"  Sous-morceau avec espace 'Ġ{m}' existe -> ID: {vocab[f'Ġ{m}']}")

if __name__ == "__main__":
    explorer_vocabulaire()