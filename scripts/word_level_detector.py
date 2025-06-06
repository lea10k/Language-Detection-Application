import json
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram


class WordLevelLanguageDetector:
    """
    Corrected word-level language detector using out-of-place scoring.
    """

    def __init__(self, model_paths: dict[str, dict[str, str]]):
        """
        Initializes the detector with the provided file paths.
        """
        # List of supported languages (e.g., ['german', 'english', 'italian'])
        self.languages = list(model_paths.keys())
        # rank_profiles[n][language] maps each n-gram to its rank integer
        self.rank_profiles: dict[int, dict[str, dict[str, int]]] = {}

        for lang, inner in model_paths.items():
            for key, filepath in inner.items():
                if not key.endswith("gram"):
                    continue
                try:
                    n = int(key.replace("gram", ""))
                except ValueError:
                    continue

                if n not in self.rank_profiles:
                    self.rank_profiles[n] = {}

                with open(filepath, "r", encoding="utf-8") as f:
                    # prob_map: { "<ngram>": <relative frequency> }
                    prob_map: dict[str, float] = json.load(f)

                # Sort n-grams by descending frequency
                sorted_ngrams = sorted(
                    prob_map.items(), key=lambda x: x[1], reverse=True
                )
                # Build a rank dictionary (rank starts at 1)
                rank_dict: dict[str, int] = {}
                for idx, (ngram_str, _) in enumerate(sorted_ngrams, start=1):
                    rank_dict[ngram_str] = idx

                # Store the rank mapping for (n, lang)
                self.rank_profiles[n][lang] = rank_dict

        # Compute penalty rank K for missing n-grams (capped at 5000)
        self.K_values: dict[int, dict[str, int]] = {}
        for n, lang2ranks in self.rank_profiles.items():
            self.K_values[n] = {}
            for lang, rankmap in lang2ranks.items():
                raw_K = len(rankmap) + 1
                self.K_values[n][lang] = min(raw_K, 5000)

    def extract_ngrams_from_word(self, word: str, n: int) -> list[str]:
        """
        Extracts n-grams from a word.
        For n=1: If the word is a padded word-unigram (starts and ends with '_'), return it;
        otherwise, extract character-level unigrams.
        """
        if n == 1:
            # If the word is padded (begins and ends with '_'), treat it as a word-unigram
            if word.startswith('_') and word.endswith('_'):
                return [word]
            else:
                # Otherwise, return individual characters
                return list(word)

        # For n > 1: extract character-level n-grams
        if len(word) < n:
            return []

        ngrams = []
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
        return ngrams

    def calculate_simple_distance(self, word: str, lang: str) -> float:
        """
        Simplified distance calculation for a single word against a language.
        """
        total_penalty = 0.0
        total_ngrams = 0

        # Weights for n-grams (1 through 5)
        weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15}

        for n in range(1, 6):
            if n not in self.rank_profiles or lang not in self.rank_profiles[n]:
                continue

            ngrams = self.extract_ngrams_from_word(word, n)
            if not ngrams:
                continue

            lang_ranks = self.rank_profiles[n][lang]
            K = self.K_values[n][lang]
            weight = weights.get(n, 0.0)

            # For each n-gram: low penalty if present in the profile,
            # high penalty (K) if not
            ngram_penalty = 0.0
            for ngram in ngrams:
                if ngram in lang_ranks:
                    # Lower rank → lower penalty
                    rank = lang_ranks[ngram]
                    penalty = rank / 1000.0  # normalized
                else:
                    penalty = K / 1000.0  # high penalty for unknown n-grams

                ngram_penalty += penalty

            # Average penalty for this n-gram level
            if ngrams:
                avg_penalty = ngram_penalty / len(ngrams)
                total_penalty += weight * avg_penalty
                total_ngrams += 1

        if total_ngrams == 0:
            return float('inf')

        return total_penalty

    def is_ambiguous_word(self, word: str, lang_scores: dict) -> bool:
        """
        Determines if a word is ambiguous based on various criteria.
        """
        # Very short words
        if len(word) <= 2:
            return True

        # Known problematic words
        problematic_words = {
            'alle', 'a', 'an', 'in', 'un', 'la', 'le', 'il', 'da', 'di',
            'we', 'my', 'it', 'so', 'to', 'me', 'is', 'am', 'go', 'no',
            'war', 'hat', 'den', 'die', 'das', 'der', 'und', 'ich', 'du',
            'e', 'i', 'o', 'u', 'per', 'con', 'del', 'che', 'non', 'una'
        }

        word_clean = word.strip('_').lower()
        if word_clean in problematic_words:
            return True

        # Words with very close scores between languages
        valid_scores = [score for score in lang_scores.values() if score != float('inf')]
        if len(valid_scores) >= 2:
            sorted_scores = sorted(valid_scores)
            best_score = sorted_scores[0]
            second_score = sorted_scores[1]

            # If the two best scores are very close
            if best_score > 0 and (second_score - best_score) / best_score < 0.3:
                return True

        return False

    def detect_single_word(self, word: str) -> dict:
        """
        Detects the language of a single word.
        """
        # Compute distances for all languages
        lang_scores = {}
        for lang in self.languages:
            score = self.calculate_simple_distance(word, lang)
            lang_scores[lang] = score

        # Debug output for selected problematic words
        if word.strip('_').lower() in ['alle', 'a', 'yesterday', 'richtig', 'abbiamo']:
            print(f"DEBUG {word}: {lang_scores}")

        # Check if the word is ambiguous
        if self.is_ambiguous_word(word, lang_scores):
            return {
                'word': word,
                'language': 'ambiguous',
                'confidence': None
            }

        # Find best and second-best language scores
        valid_scores = [(lang, score) for lang, score in lang_scores.items()
                        if score != float('inf')]

        if not valid_scores:
            return {
                'word': word,
                'language': 'ambiguous',
                'confidence': None
            }

        sorted_scores = sorted(valid_scores, key=lambda x: x[1])
        best_lang, best_score = sorted_scores[0]

        if len(sorted_scores) == 1:
            confidence = 0.7
        else:
            second_score = sorted_scores[1][1]

            # Confidence based on ratio of scores
            if second_score > best_score:
                confidence = (second_score - best_score) / second_score
                confidence = max(0.0, min(0.95, confidence))
            else:
                confidence = 0.7

        return {
            'word': word,
            'language': best_lang,
            'confidence': round(confidence, 2)
        }

    def detect_text_languages(self, text: str, window_size: int = 5) -> list[dict]:
        """
        Determines the most likely language for each word in the given text.
        """
        tokens = tokenizeWithPadding(text)
        n_tokens = len(tokens)

        # First phase: single-word detection
        preliminary = []
        for token in tokens:
            result = self.detect_single_word(token)
            preliminary.append(result)

        # Second phase: context-based refinement for ambiguous words
        final_results = [entry.copy() for entry in preliminary]

        # Enhanced context processing
        for i, entry in enumerate(preliminary):
            if entry['language'] != 'ambiguous':
                # Also check for strong context contradictions for non-ambiguous words
                word_clean = entry['word'].strip('_').lower()
                if word_clean in ['alle', 'a', 'war', 'hat', 'den', 'die', 'das', 'der']:
                    # Consider extended context
                    left = max(0, i - 4)
                    right = min(n_tokens, i + 5)
                    context_span = preliminary[left:i] + preliminary[i+1:right]

                    # Weighted context voting
                    lang_weights = {}
                    for j, ctx in enumerate(context_span):
                        ctx_lang = ctx['language']
                        if ctx_lang and ctx_lang != 'ambiguous':
                            # Closer words have higher weight
                            distance = abs(j - (i - left))
                            weight = 1.0 / (1.0 + distance * 0.5)
                            conf = ctx.get('confidence', 0.5) or 0.5
                            lang_weights[ctx_lang] = lang_weights.get(ctx_lang, 0.0) + weight * conf

                    if lang_weights:
                        best_context_lang = max(lang_weights.items(), key=lambda x: x[1])[0]
                        context_strength = lang_weights[best_context_lang]

                        # If the context is strong enough and differs from initial detection
                        if context_strength >= 2.0 and best_context_lang != entry['language']:
                            final_results[i]['language'] = best_context_lang
                            final_results[i]['confidence'] = min(0.8, 0.5 + context_strength * 0.1)
                continue

            # For ambiguous words: perform extended context analysis
            left = max(0, i - 3)
            right = min(n_tokens, i + 4)
            context_span = preliminary[left:i] + preliminary[i+1:right]

            # Weighted context voting
            lang_weights = {}
            for j, ctx in enumerate(context_span):
                ctx_lang = ctx['language']
                if ctx_lang and ctx_lang != 'ambiguous':
                    # Closer words have higher weight
                    distance = abs(j - (i - left))
                    weight = 1.0 / (1.0 + distance * 0.3)
                    conf = ctx.get('confidence', 0.5) or 0.5
                    lang_weights[ctx_lang] = lang_weights.get(ctx_lang, 0.0) + weight * conf

            if lang_weights:
                best_context_lang = max(lang_weights.items(), key=lambda x: x[1])[0]
                context_strength = lang_weights[best_context_lang]

                # Lower threshold for ambiguous words
                if context_strength >= 0.8:
                    final_results[i]['language'] = best_context_lang
                    final_results[i]['confidence'] = min(0.7, 0.3 + context_strength * 0.2)

        return final_results

