from collections import defaultdict

def calculate_language_votes(context_words: list, is_valid_vote) -> dict:
    """
    Calculate confidence votes for each language based on context words.
    
    For each context word that is not Ambiguous or unknown, 
    add its confidence to the given language vote.
    
    Args:
        context_words: List of context word detection results
        
    Returns:
        Dictionary mapping language names to their total confidence scores
        
        Example:
        If context = [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                        {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.85}]
        then language_votes = {'English': 0.95, 'Italian': 0.85}
    """
    language_votes = defaultdict(float)
    
    for word_result in context_words:
        if is_valid_vote(word_result):
            language = word_result['language']
            confidence = word_result['confidence']
            language_votes[language] += confidence
    
    return dict(language_votes)
    

def should_apply_smoothing(language_votes: dict, min_confidence_threshold: float) -> bool:
    """
    Determine if smoothing should be applied based on vote confidence.
    
    If there are any votes, determine the best language based on the highest confidence.
    If the best language has a confidence of at least MIN_CONFIDENCE_THRESHOLD, 
    smoothing should be applied.
    
    Args:
        language_votes: Dictionary of language confidence scores
        
    Returns:
        True if the best language has sufficient confidence for smoothing
    """
    if not language_votes:
        return False
    highest_confidence = max(language_votes.values())
    return highest_confidence >= min_confidence_threshold


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