import math
#import json
from scripts.n_gram import compute_n_gram
"""import tokenization
import n_gram_frequencies
from nltk.corpus import PlaintextCorpusReader"""


def calculate_perplexity(test_ngrams, model_probs, n, lambda_interp=0.9):
    """
    Calculates the perplexity of a list of n-grams against a language model.
    Uses Jelinek-Mercer interpolation smoothing:
        P_interp = λ * P_model + (1 − λ) * (1/|V|)
    where |V| is the vocabulary size (number of unique n-grams in the model).

    Args:
        test_ngrams: List of n-grams (strings) to evaluate
        model_probs: Dict mapping n-grams to their probabilities
        n: Order of the n-gram model (e.g., 2 for bigrams)
        lambda_interp: Interpolation weight (0.9 = 90% model probability)

    Returns:
        Perplexity score (float). Lower = better fit.
    """
    if not test_ngrams:
        return float('inf')

    vocab_size = len(model_probs)
    uniform_prob = 1.0 / vocab_size if vocab_size > 0 else 0.0

    log_prob_sum = 0.0
    for gram in test_ngrams:
        # Get model probability or 0 if n-gram unknown
        p_model = model_probs.get(gram, 0.0)
        
        # Apply interpolation smoothing
        p_smoothed = lambda_interp * p_model + (1 - lambda_interp) * uniform_prob
        
        # Avoid log(0) by clipping to a small value
        p_smoothed = max(p_smoothed, 1e-10)
        
        log_prob_sum += math.log2(p_smoothed)

    avg_log_prob = log_prob_sum / len(test_ngrams)
    return 2 ** (-avg_log_prob)



"""with open("Language-Detection-Application/json_data/german/3grams.json") as f:
    model_3gram = json.load(f)

with open("Language-Detection-Application/json_data/german/4grams.json") as f:
    model_4gram = json.load(f)

corpus_root = 'Language-Detection-Application/data/development/german'

corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

text = corpus.raw('dev1.txt')
tokenized_test = tokenization.tokenizeWithPadding(text)

perplexity_3 = calculate_perplexity(tokenized_test, model_3gram, n=3)
perplexity_4 = calculate_perplexity(tokenized_test, model_4gram, n=4)

print("3-Gram Perplexity:", perplexity_3)
print("4-Gram Perplexity:", perplexity_4)"""

