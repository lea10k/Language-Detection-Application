def compute_n_gram(words, n):
    """
    Generate n-grams from the input list of padded words.
    :param words: Liste von Wörtern (mit Padding, z.B. "_hallo_")
    :param n: Größe des n-grams
    :return: Liste aller n-grams (Strings)
    """
    n_grams = []
    for word in words:
        # Für jedes Wort alle n-grams extrahieren
        n_grams.extend([''.join(word[i:i + n]) for i in range(len(word) - n + 1)])
    return n_grams


"""import tokenization

print(compute_n_gram(tokenization.tokenizeWithPadding("This is a test email: test@example.com and a URL: https://www.example.com"), 4))"""
#print(compute_n_gram(['_hallo_', '_welt_'], 2))  # Example usage, should print ['_ha', 'al', 'll', 'lo', 'o_', '_w', 'we', 'el', 'lt', 't_']