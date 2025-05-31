import tokenization

def compute_n_gram(words, n):
    """
    Generate n-grams from the input text.
    
    :param text: Input text to generate n-grams from
    :type text: str
    :param n: The size of the n-grams to generate
    :type n: int
    :returns:
        list: A list of n-grams extracted from the text.
    """
    n_grams = []

    for word in words:
        n_grams.extend([''.join(word[i:i+n]) for i in range(len(word)-n+1)])

    return n_grams

#print(compute_n_gram(tokenization.tokenizeWithPadding("This is a test email: test@example.com and a URL: https://www.example.com"), 3))