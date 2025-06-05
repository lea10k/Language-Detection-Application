import json
from scripts.tokenization import tokenizeWithPadding
from scripts.n_gram import compute_n_gram
from scripts.perplexity import calculate_perplexity

class WordLevelLanguageDetector:
    def __init__(self, model_paths):
        self.models = {}
        for lang, paths in model_paths.items():
            self.models[lang] = {}
            for n, path in paths.items():
                with open(path) as f:
                    self.models[lang][n] = json.load(f)

    def detect_text_languages(self, text, window_size=5):
        tokens = tokenizeWithPadding(text)
        half = window_size // 2
        results = []

        for i in range(len(tokens)):
            start = max(0, i - half)
            end = min(len(tokens), i + half + 1)
            window = tokens[start:end]

            ngrams_3 = compute_n_gram(window, 3)
            ngrams_4 = compute_n_gram(window, 4)

            perplexities = {}
            for lang, model in self.models.items():
                p3 = calculate_perplexity(ngrams_3, model['3gram'], 3)
                p4 = calculate_perplexity(ngrams_4, model['4gram'], 4)
                perplexities[lang] = (p3 + p4) / 2  # simple average

            best_lang = min(perplexities, key=perplexities.get)
            confidence = round(perplexities[best_lang], 4)

            results.append({
                'word': tokens[i],
                'language': best_lang,
                'confidence': confidence
            })

        return results
