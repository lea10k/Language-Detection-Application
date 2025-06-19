def create_rank_map(sorted_items):
    """
    Helper function to create a rank map from sorted items.
                
    - enumerate(sorted_items, start=1) -> Gives for each n-gram its rank starting from 1; example: (1, ('al', 55)), (2, ('_ha', 38)), ...
    - for rank, (gram,_) in ... -> (gram, _) is taking only the n-gram, not the frequency. _ ignores the frequency value.
    - {gram: rank for ...} -> Building a distionary: gram = key and rank = value
    - example: rank_map = {'al': 1, '_ha': 2, ...}
    
    :param sorted_items: List of tuples where each tuple contains an n-gram and its frequency, sorted by frequency in descending order.
    :start n-gram rank at 1.
    :returns: A dictionary mapping each n-gram to its rank.
    """
    return {gram: rank for rank, (gram, _) in enumerate(sorted_items, start=1)}

def compute_distance(ranks, grams, K):
    """
    Compute the distance between the n-grams of a text and the n-gram profile of a language.
    
    Example:
    If the ranks dictionary is {'al': 1, '_ha': 2, ...} and grams is ['al', '_ha', 'xyz'],
    the function will return 1 + 2 + K (where K is the penalty rank for 'xyz' since it is not found in ranks).
    
    :param ranks: A dictionary mapping n-grams to their ranks in the language profile.
    :param grams: A list of n-grams extracted from the text.
    :param K: The penalty rank for n-grams not found in the profile.
    :returns: The sum of ranks for the n-grams in grams, using K for any n-gram not found in ranks.
    """
    return sum(ranks.get(g, K) for g in grams)

def distance_for_all_languages(word, languages, out_of_place_distance_func):
    """
    Computes the Out-of-Place distance for a word against all languages.
    
    Example:
    If the word is "hello" and languages are ['english', 'german', 'italian']:
    This function will return a dictionary like:
    {
        'english': 0.5,
        'german': 0.7,
        'italian': 0.6
    }
    
    :param word: The word to analyze.
    :param languages: A list of languages to compare against.
    :param out_of_place_distance_func: A function that computes the Out-of-Place distance for a word in a given language.
    :returns: A dictionary mapping each language to its Out-of-Place distance for the given word.
    """
    return {lang: out_of_place_distance_func(word, lang) for lang in languages}

def compute_second_best_language(sorted_langs):
    """
    Compute the second best language based on the sorted list of languages and their distances.
    :param sorted_langs: A list of tuples where each tuple contains a language and its distance.
                         Example: [('english', 0.5), ('italian', 0.6), ('german', 0.7)]
    :returns: A tuple containing the second best language and its distance.
    If there is no second best language, returns (None, float('inf')).
    """
    return sorted_langs[1] if len(sorted_langs) > 1 else (None, float('inf'))

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

def is_ambiguous(best_score, second_score, ambiguity_margin=0.25):
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


def detect_language_for_each_word(detect_Word_func, words):
    """
    Detect the language for each word in a list using the provided detection function.
    :param words: A list of words to detect the language for.
    :param detect_Word_func: A function that takes a word and returns its detected language.
    
    Example:
    If detect_Word_func is a function that detects the language of a word,
    and words is ['hello', 'welt', 'ciao'], this function will return a list like:
    [
        {'word': 'hello', 'language': 'english', 'confidence': 0.95},
        {'word': 'welt', 'language': 'german', 'confidence': 0.85},
        {'word': 'ciao', 'language': 'italian', 'confidence': 0.90}
    ]
    """
    return [detect_Word_func(token) for token in words]