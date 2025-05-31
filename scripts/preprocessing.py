import re

"""
Preprocess text for trilingual language detection (English, German, Italian).
This function removes punctuation, standalone numbers, and keeps only letters relevant for
EN/DE/IT languages, including apostrophes and spaces.
It also converts the text to lowercase and cleans up whitespace.

:param text: Input text to preprocess
:type text: str
:returns:
    str: Preprocessed text ready for language detection.
"""
def preprocessForTrilingualDetection(text):
    # Normalize spacing before punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
    
    # Replace punctuation with underscores (padding)
    text = re.sub(r'[.!?,:;@]', '_', text)
    
    # Remove typographic quotes (but keep apostrophes like it's, l'homme)
    text = text.replace('"', '').replace('“', '').replace('”', '')
    text = text.replace('‘', '').replace('’', '')

    # Keep only letters, apostrophes, underscores, spaces
    text = re.sub(r'[^a-zA-ZäöüßàèéìíòóùúçÀÈÉÌÍÒÓÙÚÇ\'_\s]', '', text)
    
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

"""print(preprocessForTrilingualDetection("This is a test email: test@example.com and a URL: https://www.example.com"))
print(preprocessForTrilingualDetection("Das ist ein Test mit Zahlen 123 und Sonderzeichen äöüß!"))
print(preprocessForTrilingualDetection("Questo è un test con numeri 456 e caratteri speciali àèéìíòóùúç!"))
print(preprocessForTrilingualDetection("Mixed languages: English, Deutsch, Italiano!"))
print(preprocessForTrilingualDetection("Lea's Haus ist schön."))
print(preprocessForTrilingualDetection("Luca's casa è bella. Ma anche la sua macchina!"))
print(preprocessForTrilingualDetection("Lea's Haus ist schön. `Luca's` casa è bella. Ma auch la sua macchina!"))
"""