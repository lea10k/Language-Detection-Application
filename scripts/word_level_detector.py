import json
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram
from perplexity import calculate_perplexity

class WordLevelLanguageDetector:
    def __init__(self, model_paths):
        # Initialize an empty dictionary to store language models
        self.models = {}
        # Iterate over each language and its corresponding n-gram model file paths
        for lang, paths in model_paths.items():
            self.models[lang] = {}
            # For each n-gram type (e.g., '3gram', '4gram') and its file path
            for n, path in paths.items():
                # Load the n-gram model from the JSON file and store it
                with open(path) as f:
                    self.models[lang][n] = json.load(f)

    def detect_text_languages(self, text, window_size=5):
        # Tokenize the input text and add padding if needed
        tokens = tokenizeWithPadding(text)
        print(f"Tokenized text: {tokens}")  # Debugging output to check tokenization
        half = window_size // 2  # Calculate half the window size for context
        results = []

        # Iterate over each token in the text
        for i in range(len(tokens)):
            # Define the window of tokens around the current token
            start = max(0, i - half)
            end = min(len(tokens), i + half + 1)
            window = tokens[start:end]
            print(f"Current window: {window}")

            # Compute 3-gram and 4-gram features for the current window
            ngrams_3 = compute_n_gram(window, 3)
            ngrams_4 = compute_n_gram(window, 4)
            print(f"3-grams: {ngrams_3}")  # Debugging output for 3-grams
            print(f"4-grams: {ngrams_4}")  # Debugging output for 4-grams

            perplexities = {}
            # Calculate perplexity for each language model
            for lang, model in self.models.items():
                p3 = calculate_perplexity(ngrams_3, model['3gram'], 3)
                p4 = calculate_perplexity(ngrams_4, model['4gram'], 4)
                perplexities[lang] = (p3 + p4) / 2  # Average the perplexities
                print(f"Perplexity for {lang}: 3-gram={p3}, 4-gram={p4}, average={perplexities[lang]}")

            # Select the language with the lowest perplexity (best match)
            best_lang = min(perplexities, key=perplexities.get)
            confidence = round(perplexities[best_lang], 4)  # Confidence as perplexity score

            # HIER MUSS ÜBERLEGT WERDEN WIE MAN DAS ERGEBNIS VISUELL IM TEXT DARSTELLEN KANN
            # Append the result for the current word
            results.append({
                'word': tokens[i],
                'language': best_lang,
                'confidence': confidence
            })

        return results
