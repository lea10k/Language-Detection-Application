import math
from tokenization import tokenize_with_padding

def compute_second_best_language(sorted_langs) -> tuple:
    """
    Compute the second best language based on the sorted list of languages and their distances.
    :param sorted_langs: A list of tuples where each tuple contains a language and its distance.
                         Example: [('english', 0.5), ('italian', 0.6), ('german', 0.7)]
    :returns: A tuple containing the second best language and its distance.
    If there is no second best language, returns (None, float('inf')).
    """
    return sorted_langs[1] if len(sorted_langs) > 1 else (None, float('inf'))


def is_ambiguous(detector, best_score, second_score, single_margin=0.1, multi_margin=0.25) -> bool:
    """
    Check if the language detection result is ambiguous based on the best and second best distance scores.
    
    args:
        detector: The language detector instance.
        best_score: The lowest (i.e., best) distance score among all candidate languages.
            This corresponds to the language profile that most closely matches the input (smaller is better).
        second_score: The second lowest distance score among the candidate languages.
            This is the next best matching language after the best.
    single_margin: The margin for ambiguity when the number of tokens is less than 10.
    multi_margin: The margin for ambiguity when the number of tokens is 10 or more.
    :returns: True if the detection is ambiguous (i.e., the top two scores are very close), False otherwise.
    """
    if math.isinf(second_score) or math.isinf(best_score):
        return False

    # Decide margin based on the number of tokens
    tokens = tokenize_with_padding(detector.input_text)
    margin = single_margin if len(tokens) < 10 else multi_margin

    rel_diff = (second_score - best_score) / best_score
    print(f"Relative difference: {rel_diff}, Margin: {margin}")
    return rel_diff < margin

def is_unknown(best_score) -> bool:
    """
    Check if the best score indicates an unknown language.
    """
    return math.isinf(best_score)
        
            