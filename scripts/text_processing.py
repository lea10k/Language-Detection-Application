import re

def preprocess_text(
    text: str,
    lowercase: bool = True,
    replace_punct_with_underscore: bool = False,
    keep_underscores: bool = False,
) -> str:
    """
    Generic text preprocessor for language detection (EN/DE/IT).
    Options:
        - lowercase: Convert text to lowercase (default: True)
        - replace_punct_with_underscore: Replace punctuation with '_' (default: False)
        - keep_underscores: Keep underscores as valid characters (default: False)
    """
    # Remove space before punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)

    # Optional: Replace punctuation with underscores (for padding)
    if replace_punct_with_underscore:
        text = re.sub(r'[.!?,:;@]', '_', text)

    # Remove typographic quotes (but keep apostrophes)
    text = (text.replace('"', '')
                .replace('“', '').replace('”', '')
                .replace('‘', '').replace('’', ''))

    # Build regex for allowed characters
    allowed = r"a-zA-ZäöüßàèéìíòóùúçÀÈÉÌÍÒÓÙÚÇ'\s"
    if keep_underscores or replace_punct_with_underscore:
        allowed += r'_'
    text = re.sub(rf"[^{allowed}]", ' ', text)

    # Optional: Replace underscores with spaces
    if not keep_underscores and replace_punct_with_underscore:
        text = text.replace('_', ' ')

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    if lowercase:
        text = text.lower()

    return text


def preprocess_for_trilingual_detection(text):
    return preprocess_text(
        text,
        lowercase=True,
        replace_punct_with_underscore=True,
        keep_underscores=False,
    )

def preprocess_for_website(text):
    return preprocess_text(
        text,
        lowercase=False,
        replace_punct_with_underscore=False,
        keep_underscores=False,
    )
