from collections import defaultdict

def compute_confidence(best_score, second_score):
    """
    Compute the confidence score based on the best and second best scores.
    The confidence score is calculated as the relative change between the second best score and the best score.
    :param best_score: The score of the best language.
    :param second_score: The score of the second best language.
    :returns: A confidence score between 0.1 and 0.95
    """
    MIN_CONFI = 0.1
    MAX_CONFI = 0.95
    return min(MAX_CONFI, max(MIN_CONFI, (second_score - best_score) / (second_score if second_score > 0 else 1)))

def calculate_language_votes(context_words: list, is_valid_vote) -> dict:
    """
    Calculate confidence votes for each language based on context words.
    
    For each context word that is not Ambiguous or unknown, 
    add its confidence to the given language vote.
    
    Args:
        context_words: List of context words of detection results
        is_valid_vote: Function to check if a word result is a valid vote. Means it is not ambiguous or unknown.
    Returns:
        Dictionary mapping language names to their total confidence scores
        
        Example:
        If context = [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                        {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.85}]
        then language_votes = {'English': 0.95, 'Italian': 0.85}
    """
    language_votes = defaultdict(float)
    
    for word in context_words:
        if is_valid_vote(word):
            language = word['language']
            confidence = word['confidence']
            language_votes[language] += confidence
    
    dict_language_votes = dict(language_votes)
    return dict_language_votes


def calculate_smoothed_confidence(vote_confidence: float, base_smooth_confi, max_smooth_confi, confi_scaling_factor) -> float:
    """
    Calculate a smoothed confidence score based on context voting.
    
    The smoothed confidence is calculated as:
    min(MAX_SMOOTHED_CONFIDENCE, BASE_SMOOTHED_CONFIDENCE + vote_confidence * SCALING_FACTOR)
    
    Args:
        vote_confidence: Total confidence from context voting
        
    Returns:
        Smoothed confidence score between 0 and MAX_SMOOTHED_CONFIDENCE
    """
    raw_confidence = base_smooth_confi + (vote_confidence * confi_scaling_factor)
    smoothed_confidence = min(max_smooth_confi, raw_confidence)
    return round(smoothed_confidence, 2)