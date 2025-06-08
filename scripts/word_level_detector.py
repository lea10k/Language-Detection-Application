import json
import detection_helper
from collections import defaultdict
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram

class WordLevelLanguageDetector:
    """
    Clearly structured language detector based on out-of-place distance.
    The profiles consist of ranked lists of n-grams.
    """
    def __init__(self, model_paths: dict[str, dict[str, str]], ngram_range=(2, 3, 4, 5)):
        self.languages = list(model_paths.keys())
        self.ngram_range = ngram_range  
        self.rank_profiles = {}  # {lang: {n: {gram: rank}}}
        self.penalty_rank = {}   # {lang: {n: K}}
        
        # Load n-gram profiles from provided paths
        for lang, ngram_files in model_paths.items():
            self.rank_profiles[lang] = {}
            """
            Example:
            model_paths = {
                'german': {}
                ...
            }
            """
            self.penalty_rank[lang] = {}
            """
            Example:
            penalty_rank = {
                'german': {},
                ...
            }
            """
            for key, path in ngram_files.items():
                if not key.endswith("gram"):
                    continue
                n = int(key[:-4])  # strip 'gram' to get the n value
                if n not in self.ngram_range:
                    continue
                
                # Load n-gram frequencies from JSON file
                with open(path, 'r', encoding='utf-8') as f:
                    freq_map = json.load(f)
                    
                # Sort n-grams by frequency and create rank mapping in descending order
                sorted_grams = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
                rank_map = detection_helper.CreateRankMap(sorted_grams)
                
                self.rank_profiles[lang][n] = rank_map
                """
                Example:
                rank_profiles = {
                    'german': {
                        2: {'al': 1, '_ha': 2, ...},
                        3: {'_ha_': 1, 'all': 2, ...},
                        ...
                    },
                    ...
                }
                """
                # Penalty-Rank: One rank higher than the highest n-gram rank
                self.penalty_rank[lang][n] = len(rank_map) + 1
                """
                Example:
                penalty_rank = {
                    'german': {
                        2: 1001,  # If a n-gram is not found, it gets this penalty rank
                        3: 1001,
                        ...
                    },
                    ...
                }
                """

    def _out_of_place_distance(self, word: str, lang: str) -> float:
        """Computes the Out-of-Place-Distance for a single word in a given language."""
        total_distance = 0.0
        ngram_counts = 0
        
        for n in self.ngram_range:
            grams = compute_n_gram([word], n)
            ranks = self.rank_profiles[lang].get(n, {})
            """
            Example: n=2
            ranks = {'al': 1, '_ha': 2, ...}
            """
            # Get the penalty rank for this n-gram size
            K = self.penalty_rank[lang].get(n, 1000)
            
            # Compute the distance for this n-gram type
            distance = detection_helper.computeDistance(ranks, grams, K)
            
            # Normalized by the number of n-grams
            total_grams = len(grams)
            if total_grams > 0:
                total_distance += distance / total_grams
                ngram_counts += 1
                
        # Average over all n-gram types (if multiple n are used)
        return total_distance / ngram_counts if ngram_counts else float('inf')

    def detect_word(self, word: str) -> dict:
        """
        Detects the language for a single (padded) word using Out-of-Place distance.
        :param word: The input word to analyze.
        :returns: A dictionary with the detected language and confidence score.
        Example:
        {
            'word': 'Hello',
            'language': 'english',
            'confidence': 0.95
        }
        If the language cannot be determined, it returns:
        {
            'word': 'Hello',
            'language': 'ambiguous',
            'confidence': None
        }
        """
        
        distances = detection_helper.distance_for_all_languages(word, self.languages, self._out_of_place_distance)
        sorted_langs = sorted(distances.items(), key=lambda kv: kv[1])
        
        # Smallest distance = best language
        best_lang, best_score = sorted_langs[0]
        # Example: best_lang = ('english', 0.1234) out of e.g. [('english', 0.1234), ('german', 0.2345), ...]
        second_lang, second_score = detection_helper.ComputeSecondBestLanguage(sorted_langs)
        
        # Ambiguous if the scores are too similar or too large
        if detection_helper.IsAmbiguous(best_score, second_score):
            return {'word': word, 'language': 'ambiguous', 'confidence': None}
        
        confidence = detection_helper.ComputeConfidence(best_score, second_score)
        return {'word': word, 'language': best_lang, 'confidence': round(confidence, 2)}


    def _apply_context_smoothing(self, results: list[dict], window: int) -> list[dict]:
        """
        Smooths the detected languages based on context.
        :param results: List of dictionaries with detected languages and confidence scores for each word.
        Example:
        [
            {'word': 'Hello', 'language': 'english', 'confidence': 0.95},
            {'word': 'Welt', 'language': 'ambiguous', 'confidence': None},
            {'word': 'Ciao', 'language': 'italian', 'confidence': 0.85},
            ...
        ]
        
        :param window: Number of words to consider for context smoothing.
        :returns: A list of dictionaries with smoothed detected languages and confidence scores for each word.
        Example:
        [
            {'word': 'Hello', 'language': 'english', 'confidence': 0.95},
            {'word': 'Welt', 'language': 'german', 'confidence': 0.85},
            ...
        ]
        
        """
        # If there are no ambiguous words or only one word, return results as is
        if len(results) <= 1:
            return results
        
        # Create a copy of results to avoid modifying the original list
        smoothed = results.copy()
        
        # Iterate over every word and check for ambiguous detections
        # If a word is ambiguous, look at the context to determine the best language
        for i, result in enumerate(results):
            if result['language'] == 'ambiguous':
                # Get context window
                start = max(0, i - window) # i = current index, window = number of words to consider before and after
                end = min(len(results), i + window + 1)
                context = results[start:i] + results[i+1:end] # context excluding the current word
                """
                Example:
                If results = [{'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                            {'word': 'Welt', 'language': 'ambiguous', 'confidence': None},
                            {'word': 'Ciao', 'language': 'italian', 'confidence': 0.85}]
                and window = 1, then context = [{'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                            {'word': 'Ciao', 'language': 'italian', 'confidence': 0.85}]"""
                
                language_votes = defaultdict(float) 
                for ctx in context:
                    # For each context word that is not ambiguous or unknown, 
                    # add its confidence to the given language vote.
                    if ctx['language'] not in ['ambiguous', 'unknown'] and ctx['confidence']:
                        language_votes[ctx['language']] += ctx['confidence']
                        """Example:
                        If context = [{'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                                      {'word': 'Ciao', 'language': 'italian', 'confidence': 0.85}]
                        then language_votes = {'english': 0.95, 'italian': 0.85}"""
                
                # If there are any votes, determine the best language based on the highest confidence
                # If the best language has a confidence of at least 1.0, assign it to the current word
                # and set a smoothed confidence score.       
                if language_votes:
                    best_lang = max(language_votes, key=language_votes.get)
                    if language_votes[best_lang] >= 1.0:
                        smoothed[i]['language'] = best_lang
                        smoothed[i]['confidence'] = min(0.8, 0.3 + language_votes[best_lang] * 0.1)
        return smoothed
    
    def detect_text_languages(self, text: str, context_window=3) -> list[dict]:
            """
            Detect the language for each word in a text using the provided detection function.
            :param text: The input text to analyze.
            :param context_window: Number of words to consider for context smoothing.
            :returns: A list of dictionaries with detected languages and confidence scores for each word.
            Example:
            [
                {'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                {'word': 'Welt', 'language': 'german', 'confidence': 0.85},
                ...
            ]
            """
            tokens = tokenizeWithPadding(text)
            results = detection_helper.DetectLanguageForEachWord(self.detect_word, tokens) 
            return self._apply_context_smoothing(results, context_window)

    def get_language_summary(self, results: list[dict]) -> dict:
        lang_counts = defaultdict(int) # count of words per language
        total_confidence = defaultdict(float) #sum of confidence scores for each language
        
        for result in results:
            lang = result['language']
            if lang not in ['ambiguous', 'unknown']:
                lang_counts[lang] += 1
                """Example:
                If results = [{'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                            {'word': 'Welt', 'language': 'german', 'confidence': 0.85},
                            {'word': 'Ciao', 'language': 'italian', 'confidence': 0.75}]
                then lang_counts = {'english': 1, 'german': 1, 'italian': 1}"""
                total_confidence[lang] += result['confidence'] or 0
                """
                Example:
                If results = [{'word': 'Hello', 'language': 'english', 'confidence': 0.95},
                            {'word': 'Welt', 'language': 'german', 'confidence': 0.85},
                            {'word': 'Ciao', 'language': 'italian', 'confidence': 0.75}]
                then total_confidence = {'english': 0.95, 'german': 0.85, 'italian': 0.75}"""

                
        summary = {}
        total_words = sum(lang_counts.values()) # total number of words detected
        for lang, count in lang_counts.items():
            avg_confidence = total_confidence[lang] / count if count > 0 else 0.0
            summary[lang] = {
                'words': count,
                'percentage': round((count / total_words) * 100, 1) if total_words > 0 else 0.0,
                'avg_confidence': round(avg_confidence, 2)
            }
        return summary
