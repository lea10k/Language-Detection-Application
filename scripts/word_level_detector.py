import json
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram


class WordLevelLanguageDetector:
    """
    Refactored word-level language detector using out-of-place scoring.
    Uses compute_n_gram for n-gram extraction.
    """

    def __init__(self, model_paths: dict[str, dict[str, str]]):
        """
        Load n-gram rank profiles and compute penalty ranks.
        """
        print("Initializing WordLevelLanguageDetector...")
        self.languages = list(model_paths.keys())
        print(f"Supported languages: {self.languages}")
        self.rank_profiles: dict[int, dict[str, dict[str, int]]] = {}
        """
        Example structure:
        rank_profiles = 
        {
            2: {
                'german': {'_h': 1, 'ha': 2, 'al': 3, ...},
                'english': {'_t': 1, 'th': 2, 'he': 3, ...},
                'italian': {'_i': 1, 'it': 2, 'ta': 3, ...}
            },
            3: {
                'german': {'_ha': 1, 'hal': 2, ...},
                'english': {'_th': 1, 'the': 2, ...},
                'italian': {'_it': 1, 'ita': 2, ...}
            }
            ...
        }
        """
        self.K_values: dict[int, dict[str, int]] = {}

        # Load rank profiles for each n and language
        for lang, paths in model_paths.items():
            for key, path in paths.items():
                if not key.endswith("gram"):  # expect keys like '2gram', '3gram'
                    continue
                n = int(key[:-4])  # strip 'gram' to get the n value
                profile = self._load_rank_profile(path) # example: {'_ha': 1, 'al': 2, ...}
                self.rank_profiles.setdefault(n, {})[lang] = profile
                """
                dictionary access: my_dict[key] = value
                .setdefault(n, {}) -> If n is not in the dictionary, set it to an empty dictionary
                my_dict.setdefault(n, {})[lang] = profile -> Adds the profile for the language under n
                """
                print(f"Loaded {n}-gram profile for {lang}: {len(profile)} entries")

        # Compute penalty rank K for missing n-grams
        for n, lang_profiles in self.rank_profiles.items():
            """
            Examaple rank_profiles.items():
            rank_profiles.items() -> [(2, {'german': {'_h': 1, 'ha': 2}, 'english': {'_t': 1, 'th': 2}}),
            (3, {'german': {'_ha': 1, 'hal': 2}, 'english': {'_th': 1, 'the': 2}})]
            """
            self.K_values[n] = {}
            """
            Example: n = 2
            K_values = {
                2: {}
            }
            """
            for lang, profile in lang_profiles.items():
                K = min(len(profile) + 1, 5000) #K is the penalty rank for missing n-grams
                """
                K = min(len(profile) + 1, 5000) -> If the number of n-grams is less than 5000, K is set to len(profile) + 1
                + 1 because we have to give unknown n-grams a rank that is worse than the worst known n-gram
                """
                self.K_values[n][lang] = K
                """
                Example: n = 2, lang = 'german'
                K_values = {
                    2: {'german': 1001, ...}
                }
                """
                print(f"Computed K for {lang} {n}-grams: {K}")

    def _load_rank_profile(self, filepath: str) -> dict[str, int]:
        """
        Load JSON of n-gram frequencies and return n-gram -> rank mapping.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            freq_map: dict[str, float] = json.load(f) # example: {'_ha': 0.0012, 'al': 0.0023, ...}
        sorted_items = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True) 
        #freq_map.items() -> list of tuples (n-gram, frequency)
        #kv[1] -> List is sorted by second item (frequency) in descending order
        #example: sorted_items = [('al', 0.0023), ('_ha', 0.0012), ...]
        rank_map = {gram: rank for rank, (gram, _) in enumerate(sorted_items, start=1)}
        #enumerate(sorted_items, start=1) -> Gives for each n-gram its rank starting from 1; example: (1, ('al', 0.0023)), (2, ('_ha', 0.0012)), ...
        #for rank, (gram,_) in ... -> (gram, _) is taking only the n-gram, not the frequency. _ ignores the frequency value.
        #{gram: rank for ...} -> Building a distionary: gram = key and rank = value
        #example: rank_map = {'al': 1, '_ha': 2, ...}
        print(f"Profile loaded from {filepath}, total {len(rank_map)} n-grams")
        return rank_map

    def _distance(self, word: str, lang: str) -> float:
        """
        Compute weighted out-of-place distance for a single word and language.
        Extracts n-grams via compute_n_gram.
        """
        weights = {1:0.1, 2:0.2, 3:0.3, 4:0.35, 5:0.15} # Weights for each n-gram level	
        total, count = 0.0, 0
        print(f"Calculating distance for '{word}' vs {lang}")

        for n, profile in self.rank_profiles.items():
            if lang not in profile:
                continue
            # Use compute_n_gram on single-word list
            grams = compute_n_gram([word], n) # Example: compute_n_gram(['_hallo_'], 2) → ['_h', 'ha', 'al', 'll', 'lo', 'o_']
            print(f"Extracted {len(grams)} {n}-grams from '{word}': {grams}")
            if not grams: #if no n-grams were created, skip this level
                continue
            ranks = profile[lang] # Example: {'_ha': 1, 'al': 2, ...}
            K = self.K_values[n][lang] # Get penalty rank for this language and n-gram size
            avg_penalty = sum((ranks.get(g, K) / 1000.0) for g in grams) / len(grams)
            """
            ranks.get(g, K) -> Get the rank of the n-gram g, or K if it is not found.
            
            Example:
            _hallo_ → compute_n_gram(['_hallo_'], 2) → ['_h', 'ha', 'al', 'll', 'lo', 'o_']
            ranks = {'_ha': 4, 'al': 10, 'll': unknown, ...} -> K = 2500 -> 2.5
            avg_penalty = (4/1000 + 10/1000 + 2500/1000) / 6
            """
            contrib = weights[n] * avg_penalty
            total += contrib
            count += 1 # Count how many n-gram levels contributed
            """
            If the word is too short, e.g. "_du_":

            Length: 4 → results in only:

            1-grams: works

            2-grams: works

            3-grams: ['_du', 'du_'] → works

            4-grams: ['_du_'] → works

            5-grams: does not work (because word < 5 characters)

            → count = 4
            """
            print(f"  level {n}: avg_penalty {avg_penalty:.4f}, weight {weights[n]} -> contrib {contrib:.4f}")

        distance = total if count > 0 else float('inf')
        print(f"Total distance for '{word}' vs {lang}: {distance:.4f}")
        return distance

    def detect_single_word(self, word: str) -> dict:
        """
        Detect language for a single padded word token.
        """
        print(f"\nDetecting single word: {word}")
        scores = {lang: self._distance(word, lang) for lang in self.languages}
        # scores shows the scores for one word in different languages
        # Example: scores = {'german': 0.1234, 'english': 0.5678, 'italian': float('inf')}
        print(f"Scores: {scores}")

        # Ambiguity check: too short or close scores
        sorted_vals = sorted(v for v in scores.values() if v != float('inf')) # Example: sorted_vals = [0.1234, 0.5678] if 'italian' is inf
        if len(sorted_vals) >= 2 and (sorted_vals[1] - sorted_vals[0]) / sorted_vals[0] < 0.3:
            print(f"'{word}' marked ambiguous: close scores {sorted_vals[0]:.4f}, {sorted_vals[1]:.4f}")
            return {'word': word, 'language': 'ambiguous', 'confidence': None}

        # Choose best and compute confidence
        # Sort scores from lowest (best) to highest
        # Example: sorted_langs = [('german', 0.1234), ('english', 0.5678), ('italian', float('inf'))]
        sorted_langs = sorted(scores.items(), key=lambda kv: kv[1])

        # Get best and second-best language-score tuples
        best_lang, best_score = sorted_langs[0] # Example: ('german', 0.1234)
        second_lang, second_score = sorted_langs[1] if len(sorted_langs) > 1 else (None, float('inf'))
        # Example: second_lang = ('english', 0.5678) if it exists, otherwise (None, float('inf'))

        # Print for debugging
        print(f"Best: {best_lang} ({best_score:.4f}), Second: {second_lang} ({second_score:.4f})")

        # Calculate confidence
        if second_score == float('inf'): # No second language found
            confidence = 0.7
        else:
            confidence = (second_score - best_score) / second_score 
            # How much better is the best score than the second?
            # Example: (0.5678 - 0.1234) / 0.5678 = 0.2172
            # Best score is 21.72% better than second score
            # the smaller, the better.
            confidence = max(0.0, min(0.95, confidence)) # Clamp confidence to [0.0, 0.95] range

        # Return structured result
        return {
            'word': word,
            'language': best_lang,
            'confidence': round(confidence, 2)
        }


    def detect_text_languages(self, text: str, window_size: int = 5) -> list[dict]:
        """
        Detect languages for each word in text and refine ambiguous via neighbors.
        """
        print(f"\nDetecting text: {text}")
        tokens = tokenizeWithPadding(text)
        # Example: tokens = ['_alle_', '_meine_', '_entchen_', '_sind_', '_hier_']
        print(f"Tokens: {tokens}")

        results = [self.detect_single_word(t) for t in tokens]
        # Example: results = [{'word': '_alle_', 'language': 'ambiguous', 'confidence': None}, 
        # {'word': '_meine_', 'language': 'german', 'confidence': 0.86}, 
        # {'word': '_entchen_', 'language': 'ambiguous', 'confidence': None}, ...]
        print(f"Preliminary results: {results}")

        # context pass for ambiguous
        for i, res in enumerate(results):
            # Example enumerate(results): [(0, {'word': '_alle_', 'language': 'ambiguous', 'confidence': None}), ...]
            if res['language'] != 'ambiguous':
                continue
            left, right = max(0, i-3), min(len(tokens), i+4) # context window of 3 words left and right
            neighbors = results[left:i] + results[i+1:right] # 
            """
            Example neighbors:
            If i = 2 (word '_entchen_'):
            neighbors = [
                {'word': '_alle_', 'language': 'ambiguous', 'confidence': None},
                {'word': '_meine_', 'language': 'german', 'confidence': 0.86}, 
                {'word': '_sind_', 'language': 'german', 'confidence': 0.75}, 
                {'word': '_hier_', 'language': 'german', 'confidence': 0.80}
            ]
            """
            print(f"Context for {res['word']}: {[n['language'] for n in neighbors]}")
            votes = {}
            for nbr in neighbors:
                lang = nbr['language']
                if lang != 'ambiguous':
                    votes[lang] = votes.get(lang, 0) + (nbr['confidence'] or 0.5)
                    # Example: votes = {'german': 1.61}
                    """
                    If e.g. german is in the votes dictionary, increase its score by the confidence of the neighbor.
                    If it is not in the votes dictionary, add it with the confidence of the neighbor or 0.5 if confidence is None.
                    """
            print(f"Votes: {votes}")
            if votes:
                best_lang, score = max(votes.items(), key=lambda kv: kv[1]) # searches for the language with the highest score
                if score >= 1.0: # if the context vote is strong enough (sum of confidence values ≥ 1.0), the word is reclassified.
                    print(f"Reassign '{res['word']}' to {best_lang} via context (score {score:.2f})")
                    res['language'] = best_lang
                    res['confidence'] = round(min(0.8, 0.3 + score*0.1),2)
                    """
                    - score*0.1 increases the confidence with more agreement
                    - + 0.3: base confidence
                    - min(0.8, ...): limit the maximum value so that context does not become "too confident"
                    """

        print(f"Final results: {results}\n")
        return results
