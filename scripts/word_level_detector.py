import json
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram
from perplexity import calculate_perplexity

class WordLevelLanguageDetector:
    # Common short words per language (with padding tokens)
    SHORT_WORDS = {
        'german': {'_der_', '_die_', '_das_', '_und_', '_zu_', '_ich_'},
        'english': {'_the_', '_and_', '_to_', '_of_', '_it_', '_is_'},
        'italian': {'_il_', '_la_', '_e_', '_di_', '_che_',  '_ma_'}
    }

    def __init__(self, model_paths):
        """
        Initialize the detector with language model paths.
        
        Args:
            model_paths: Dict mapping language codes to dicts of n-gram paths.
                Example: {'en': {'2gram': 'path/to/en_2grams.json', ...}}
        """
        self.models = {}
        # Load n-gram models for each language and n-gram size
        for lang, paths in model_paths.items():
            self.models[lang] = {}
            for n, path in paths.items():
                with open(path) as f:
                    self.models[lang][n] = json.load(f)

        self.min_token_length = 4  # Minimum token length to analyze

    def get_ngram_weights(self, token_length):
        """
        Assign dynamic weights to n-gram sizes based on token length.
        
        Args:
            token_length: Length of the token.
        Returns:
            Tuple of weights for 2, 3, 4, and 5-grams.
        """
        if token_length <= 4:
            return (0.4, 0.4, 0.2, 0.0)  # Emphasize 2-3 grams for short words
        elif token_length <= 8:
            return (0.2, 0.3, 0.3, 0.2)
        else:
            return (0.1, 0.2, 0.4, 0.3)  # Favor longer n-grams for long words

    def detect_text_languages(self, text, window_size=5):
        """
        Detect language for each token in the input text.
        
        Args:
            text: Input string (potentially multilingual)
            window_size: Context window size (in tokens)
            
        Returns:
            List of dicts with keys: 'word', 'language', 'confidence'
        """
        tokens = tokenizeWithPadding(text)
        half_window = window_size // 2
        results = []

        # First pass: initial classification
        for i, token in enumerate(tokens):
            # Handle short tokens using predefined word lists
            if len(token) <= self.min_token_length:
                lang = next((l for l, words in self.SHORT_WORDS.items() 
                             if token.lower() in words), 'undetermined')
                results.append({
                    'word': token,
                    'language': lang,
                    'confidence': 1.0 if lang != 'undetermined' else None
                })
                continue

            # Get context window around the token
            start = max(0, i - half_window)
            end = min(len(tokens), i + half_window + 1)
            window = tokens[start:end]

            # Compute n-grams for the window
            ngrams = {
                2: compute_n_gram(window, 2),
                3: compute_n_gram(window, 3),
                4: compute_n_gram(window, 4),
                5: compute_n_gram(window, 5)
            }

            # Get dynamic weights for n-gram sizes
            weights = self.get_ngram_weights(len(token))

            # Calculate weighted perplexity for each language
            lang_scores = {}
            for lang, models in self.models.items():
                score = 0.0
                for n, weight in zip([2, 3, 4, 5], weights):
                    if weight > 0:
                        model = models.get(f"{n}gram", {})
                        perplexity = calculate_perplexity(ngrams[n], model, n)
                        score += weight * perplexity
                lang_scores[lang] = score

            # Normalize scores to 0-1 range (lower is better)
            if lang_scores:
                min_score = min(lang_scores.values())
                max_score = max(lang_scores.values())
                score_range = max_score - min_score if max_score != min_score else 1.0
                
                normalized_scores = {
                    lang: (score - min_score) / score_range
                    for lang, score in lang_scores.items()
                }
                
                best_lang = min(normalized_scores, key=normalized_scores.get)
                confidence = 1 - normalized_scores[best_lang]
                
                if confidence > 0.5:
                    results.append({
                        'word': token,
                        'language': best_lang,
                        'confidence': round(confidence, 2)
                    })
                else:
                    results.append({
                        'word': token,
                        'language': 'undetermined',
                        'confidence': None
                    })
            else:
                results.append({
                    'word': token,
                    'language': 'undetermined',
                    'confidence': None
                })

        # Second pass: context-based disambiguation
        for i in range(len(results)):
            if results[i]['language'] == 'undetermined':
                # Check 3 tokens before and after
                context = results[max(0,i-3):min(len(results),i+4)]
                
                # Count language occurrences in context
                lang_counts = {}
                for entry in context:
                    lang = entry['language']
                    if lang != 'undetermined':
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                # Apply context override if strong evidence
                if lang_counts:
                    best_lang, count = max(lang_counts.items(), key=lambda x: x[1])
                    if count >= 2:  # Require at least 2 agreeing neighbors
                        results[i]['language'] = best_lang
                        results[i]['confidence'] = 0.5  # Moderate confidence

        return results