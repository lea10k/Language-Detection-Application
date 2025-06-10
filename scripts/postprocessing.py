import re

def PostprocessText(text):
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
    # Normalize spacing before punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
    
    # Remove typographic quotes (but keep apostrophes like it's, l'homme)
    text = text.replace('"', '').replace('“', '').replace('”', '')
    text = text.replace('‘', '').replace('’', '')

    # Keep only letters, apostrophes, underscores, spaces
    text = re.sub(r'[^a-zA-ZäöüßàèéìíòóùúçÀÈÉÌÍÒÓÙÚÇ\'_\s]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def ReplaceProcessedText(results: list[dict], input_text: str) -> dict:
    processed_text = PostprocessText(input_text).split()
    for i, item in enumerate(results):
        item["word"] = processed_text[i] if i < len(processed_text) else item["word"]
    return results


language_colors = {
    "English": "#FF0000",
    "German": "#2DDF00",
    "Italian": "#0000FF",
    "Ambiguous": "#CCCCCC",
}

def colorize_text(results: dict) -> str:
    colored = []
    for item in results:
        word = item["word"]
        color = language_colors.get(item["language"], "#000000")
        # Erzeuge ein span-Element mit Inline-Style
        colored.append(f'<span style="color: {color};">{word}</span>')
    return " ".join(colored)