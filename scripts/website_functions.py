from text_processing import preprocess_for_website
import numpy as np

def replace_processed_text(results: list[dict], input_text: str) -> dict:
    processed_text = preprocess_for_website(input_text).split()
    for i, item in enumerate(results):
        item["word"] = processed_text[i] if i < len(processed_text) else item["word"]
    return results

language_colors = {
    "English": "#FF0000",
    "German": "#2DDF00",
    "Italian": "#0000FF",
    "Ambiguous": "#676767",
}

def colorize_text(results: dict) -> str:
    colored = []
    for item in results:
        word = item["word"]
        color = language_colors.get(item["language"], "#000000")
        colored.append(f'<span style="color: {color};">{word}</span>')
    return " ".join(colored)

def count_amount_of_words_of_language(results: list) -> dict:
    """
    Count the number of words detected for each language.
    
    Args:
        results: List of word detection results
        
    Returns:
        Dictionary mapping language names to word counts
    """
    languages = extract_languages_from_results(results)
    unique_languages, counts = np.unique(languages, return_counts=True)
    return dict(zip(unique_languages, counts))

def extract_languages_from_results(results: list) -> np.ndarray:
    """
    Extract language labels from detection results.
    
    Args:
        results: List of detection result dictionaries
        
    Returns:
        NumPy array of language labels
    """
    languages = np.array([])
    for result in results:
        language = result.get("language")
        languages = np.append(languages, language)
    return languages

def percentage_of_language(language_word_counts: dict) -> dict:
    """
    Calculate percentage distribution of languages.
    
    Args:
        language_word_counts: Dictionary mapping languages to word counts
        
    Returns:
        Dictionary mapping languages to percentage strings (e.g., "25.5%")
    """
    total_words = sum(language_word_counts.values())
    
    percentage_results = {}
    for language, word_count in language_word_counts.items():
        percentage = calculate_percentage(word_count, total_words)
        percentage_results[language] = f"{percentage}%"
    
    return percentage_results
    
def calculate_percentage(part: int, total: int) -> float:
    """
    Calculate percentage and round to 2 decimal places.
    
    Args:
        part: Partial count (numerator)
        total: Total count (denominator)
        
    Returns:
        Percentage value rounded to 2 decimal places
    """
    percentage = (part / total) * 100
    return round(percentage, 2)