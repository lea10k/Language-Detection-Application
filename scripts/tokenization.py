from preprocessing import preprocessForTrilingualDetection

def tokenizeWithPadding(text):
    """
    Tokenizes the input text after preprocessing for trilingual detection, and adds padding underscores to each word.

    :args:
        text (str): The input text to be tokenized and padded.
    :returns:
        list: A list of words from the input text, each padded with an underscore at the beginning and end.
    Example:
        >>> tokenizeWithPadding("Hello world!")
        ['_hello_', '_world_']
    Note:
        The input text is first preprocessed using `preprocessForTrilingualDetection` from the `preprocessing` module.
    """
    text = preprocessForTrilingualDetection(text) 
    words = text.split()
    # Add underscores at the beginning and end of each word
    padded_words = ["_" + word + "_" for word in words if word.strip()]
    
    return padded_words

"""print(tokenizeWithPadding("This is a test email: test@example.com and a URL: https://www.example.com"))
print(tokenizeWithPadding("Das ist ein Test mit Zahlen 123 und Sonderzeichen äöüß!"))
print(tokenizeWithPadding("Questo è un test con numeri 456 e caratteri speciali àèéìíòóùúç!"))
print(tokenizeWithPadding("Mixed languages: English, Deutsch, Italiano!"))
print(tokenizeWithPadding("Lea's Haus ist schön."))
print(tokenizeWithPadding("Luca's casa è bella. Ma anche la sua macchina!"))"""