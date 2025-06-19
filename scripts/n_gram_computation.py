import numpy as np

def compute_n_gram(words, n):
    """
    Generate n-grams from the input list of padded words.
    :param words: List of words (with padding, e.g., "_hallo_")
    :param n: Size of the n-gram
    :return: List of all n-grams (strings)
    """
    n_grams = []
    for word in words:
        arr = np.array([word[i:i + n] for i in range(len(word) - n + 1)])
        n_grams.extend(arr)
    return np.array(n_grams)
    


"""import tokenization

print(compute_n_gram(tokenization.tokenizeWithPadding("This is a test email: test@example.com and a URL: https://www.example.com"), 4))"""
#print(compute_n_gram(['_hallo_', '_welt_'], 2))  # Example usage, should print ['_ha', 'al', 'll', 'lo', 'o_', '_w', 'we', 'el', 'lt', 't_']