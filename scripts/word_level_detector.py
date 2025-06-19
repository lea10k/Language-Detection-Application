from typing import Dict, List, Union, Tuple
import detection_helper
from tokenization import tokenizeWithPadding

from language_model_loader import LanguageModelLoader
from language_distance_calculator import LanguageDistanceCalculator
from context_smoothing import ContextSmoother

class WordLevelLanguageDetector(LanguageModelLoader, LanguageDistanceCalculator, ContextSmoother):
    """
    Orchestrates language detection using models, distances, and smoothing.
    """
    DEFAULT_NGRAM_RANGE = (2, 3, 4, 5)
    DEFAULT_CONTEXT_WINDOW = 3
    AMBIGUOUS_LANGUAGE = 'Ambiguous'
    UNKNOWN_LANGUAGE = 'unknown'

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
        distances = self.compute_all_language_distances(word)
        sorted_langs = self.sort_languages_by_distance(distances)
        best, best_score = sorted_langs[0]
        second, second_score = detection_helper.compute_second_best_language(sorted_langs)
        if detection_helper.is_ambiguous(best_score, second_score):
            return {'word': word, 'language': self.AMBIGUOUS_LANGUAGE, 'confidence': None}
        conf = detection_helper.compute_confidence(best_score, second_score)
        return {'word': word, 'language': best, 'confidence': round(conf, 2)}

    def detect_text_languages(self, text: str,
                              context_window: int = DEFAULT_CONTEXT_WINDOW) -> List[Dict]:
        tokens = tokenizeWithPadding(text)
        results = detection_helper.detect_language_for_each_word(self.detect_word, tokens)
        return self.apply_context_smoothing(results, context_window)
