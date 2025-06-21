from typing import Dict, List, Union, Tuple
import detection_helper
from tokenization import tokenizeWithPadding

from language_model_loader import LanguageModelLoader
from language_distance_calculator import LanguageDistanceCalculator
from context_smoothing import ContextSmoother
from confidence import compute_confidence

class WordLevelLanguageDetector(LanguageModelLoader, LanguageDistanceCalculator, ContextSmoother):
    """
    Orchestrates language detection using models, distances, and smoothing.
    """
    DEFAULT_NGRAM_RANGE = (2, 3, 4, 5)
    DEFAULT_CONTEXT_WINDOW = 3
    AMBIGUOUS_LANGUAGE = 'Ambiguous'
    UNKNOWN_LANGUAGE = 'Unknown'

    MIN_CONFIDENCE_THRESHOLD = 1.0
    BASE_SMOOTHED_CONFIDENCE = 0.3
    CONFIDENCE_SCALING_FACTOR = 0.1
    MAX_SMOOTHED_CONFIDENCE = 0.8

    def __init__(self, model_paths: Dict[str, Dict[str, str]],
                 ngram_range: Tuple[int, ...] = DEFAULT_NGRAM_RANGE):
        self.languages = list(model_paths.keys())
        self.ngram_range = ngram_range
        self.load_language_models(model_paths)

    def detect_word(self, word: str) -> Dict[str, Union[str, float, None]]:
        """
        Detect the language of a single word.
        Args:
            word (str): The word to detect the language for.
        Returns:
            Dict: A dictionary containing the word, detected language, and confidence score.
        Example:
            If word = "Hello", the output might be:
            {
                'word': 'Hello',
                'language': 'English',
                'confidence': 0.95
            }
        """
        distances = self.compute_all_language_distances(word)
        sorted_langs = self.sort_languages_by_distance(distances)
        #print(f"Sorted languages for word '{word}': {sorted_langs}")
        
        best_lang, best_score = sorted_langs[0]
        second_lang, second_score = detection_helper.compute_second_best_language(sorted_langs)
        
        if detection_helper.is_unknown(best_score):
            return {'word': word, 'language': self.UNKNOWN_LANGUAGE, 'confidence': None}
        
        if detection_helper.is_ambiguous(best_score, second_score):
            return {'word': word, 'language': self.AMBIGUOUS_LANGUAGE, 'confidence': None}
        
        conf = compute_confidence(best_score, second_score)
        return {'word': word, 'language': best_lang, 'confidence': round(conf, 2)}

    def detect_text_languages(self, text: str,
                              context_window: int = DEFAULT_CONTEXT_WINDOW) -> List[Dict]:
        """
        Detect languages for each word in a given text, applying context smoothing.
        Args:
            text (str): The input text to analyze.
            context_window (int): The number of surrounding words to consider for smoothing.
        Returns:
            List[Dict]: A list of dictionaries, each containing the word, detected language, and confidence score.
        Example:
            If text = "Hello world", the output might be:
            [
                {'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                {'word': 'world', 'language': 'English', 'confidence': 0.90}
            ]
        """
        input_tokens = tokenizeWithPadding(text)
        results_for_each_input_word = [self.detect_word(token) for token in input_tokens]
        return self.apply_context_smoothing(results_for_each_input_word, context_window)
