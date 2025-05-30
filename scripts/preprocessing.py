import re
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
    
def preprocessForTrilingualDetection(text):
    # Remove emails, URLs, numbers
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    
    # Keep only letters relevant for EN/DE/IT + apostrophes + spaces
    text = re.sub(r'[^a-zA-ZäöüßàèéìíòóùúçÀÈÉÌÍÒÓÙÚÇ\'\s]', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

print(preprocessForTrilingualDetection("This is a test email: test@example.com and a URL: https://www.example.com"))
print(preprocessForTrilingualDetection("Das ist ein Test mit Zahlen 123 und Sonderzeichen äöüß!"))
print(preprocessForTrilingualDetection("Questo è un test con numeri 456 e caratteri speciali àèéìíòóùúç!"))
print(preprocessForTrilingualDetection("Mixed languages: English, Deutsch, Italiano!"))
print(preprocessForTrilingualDetection("Lea's Haus ist schön."))