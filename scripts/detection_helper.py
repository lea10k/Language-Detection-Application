import math

def compute_second_best_language(sorted_langs) -> tuple:
    """
    Compute the second best language based on the sorted list of languages and their distances.
    :param sorted_langs: A list of tuples where each tuple contains a language and its distance.
                         Example: [('english', 0.5), ('italian', 0.6), ('german', 0.7)]
    :returns: A tuple containing the second best language and its distance.
    If there is no second best language, returns (None, float('inf')).
    """
    return sorted_langs[1] if len(sorted_langs) > 1 else (None, float('inf'))


def is_ambiguous(best_score, second_score, ambiguity_margin=0.25) -> bool:
    """
    Check if the language detection result is ambiguous based on the best and second best distance scores.

    :param best_score: The lowest (i.e., best) distance score among all candidate languages.
        This corresponds to the language profile that most closely matches the input (smaller is better).
    :param second_score: The second lowest distance score among the candidate languages.
        This is the next best matching language after the best.
    :param ambiguity_margin: The threshold for ambiguity, given as a relative margin (default: 0.25).
        If the relative difference between the best and second-best scores is smaller than this margin,
        the detection is considered ambiguous.
    :returns: True if the detection is ambiguous (i.e., the top two scores are very close), False otherwise.
    """
    if second_score < float('inf') and (second_score - best_score) / best_score < ambiguity_margin:
        return True
    return False

def is_unknown(best_score) -> bool:
    """
    Check if the best score indicates an unknown language.
    """
    if math.isinf(best_score):
        return True
    return False
        
            