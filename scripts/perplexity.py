import math
import json
from n_gram import compute_n_gram
import tokenization
import n_gram_frequencies

def calculate_perplexity(test_tokens, model_probs, n, smoothing=1e-4):
    n_grams = compute_n_gram(test_tokens, n)
    log_prob_sum = 0
    N = len(n_grams)

    for gram in n_grams:
        prob = model_probs.get(gram, smoothing)  # Laplace smoothing
        log_prob_sum += math.log2(prob)

    return 2 ** (-log_prob_sum / N)

with open("Language-Detection-Application/json_data/german/3grams.json") as f:
    model_3gram = json.load(f)

with open("Language-Detection-Application/json_data/german/4grams.json") as f:
    model_4gram = json.load(f)



tokenized_test = tokenization.tokenizeWithPadding('Language-Detection-Application/data/development/german/dev1.txt')

perplexity_3 = calculate_perplexity(tokenized_test, model_3gram, n=3)
perplexity_4 = calculate_perplexity(tokenized_test, model_4gram, n=4)

print("3-Gram Perplexity:", perplexity_3)
print("4-Gram Perplexity:", perplexity_4)
