import json
import detection_helper
import numpy as np
import confidence
from tokenization import tokenizeWithPadding
from n_gram_computation import compute_n_gram
from typing import Dict, List, Tuple, Optional, Union

class WordLevelLanguageDetector:
    """
    Clearly structured language detector based on out-of-place distance.
    The profiles consist of ranked lists of n-grams.
    
    The detector works by:
    1. Loading n-gram frequency profiles for each language
    2. Computing out-of-place distances for words against each language profile
    3. Selecting the language with the smallest distance
    4. Applying context smoothing to resolve ambiguous detections
    """
    
    # Constants for configuration
    DEFAULT_NGRAM_RANGE = (2, 3, 4, 5)
    DEFAULT_CONTEXT_WINDOW = 3
    AMBIGUOUS_LANGUAGE = 'Ambiguous'
    UNKNOWN_LANGUAGE = 'unknown'
    
    # Context smoothing constants
    MIN_CONFIDENCE_THRESHOLD = 1.0
    BASE_SMOOTHED_CONFIDENCE = 0.3
    CONFIDENCE_SCALING_FACTOR = 0.1
    MAX_SMOOTHED_CONFIDENCE = 0.8
    
    #--- Initialization ---
    
    def __init__(self, model_paths: Dict[str, Dict[str, str]], ngram_range: Tuple[int, ...] = DEFAULT_NGRAM_RANGE):
        """
        Initialize the language detector with n-gram models.
        
        Args:
            model_paths: Dictionary mapping language names to their n-gram model file paths
                        Example:
                        {
                            'German': {'2gram': 'path/to/german_2gram.json', '3gram': 'path/to/german_3gram.json'},
                            'English': {'2gram': 'path/to/english_2gram.json', '3gram': 'path/to/english_3gram.json'},
                            ...
                        }
            ngram_range: Tuple of n-gram sizes to use for detection
        """
        self.languages = list(model_paths.keys())
        self.ngram_range = ngram_range
        
        # Initialize data structures for n-gram profiles
        self.rank_profiles = {}  # {lang: {n: {gram: rank}}}
        """
        Example:
        rank_profiles = {
            'German': {
                2: {'al': 1, '_ha': 2, ...},
                3: {'_ha_': 1, 'all': 2, ...},
                ...
            },
            'English': {
                2: {'th': 1, 'he': 2, ...},
                3: {'the': 1, 'and': 2, ...},
                ...
            },
            ...
        }
        """
        
        self.penalty_rank = {}   # {lang: {n: penalty_value}}
        """
        Example:
        penalty_rank = {
            'German': {
                2: 1001,  # If a n-gram is not found, it gets this penalty rank
                3: 1001,
                ...
            },
            'English': {
                2: 1001,
                3: 1001,
                ...
            },
            ...
        }
        """
        
        # Load n-gram profiles from provided paths
        self.load_language_models(model_paths)
    
    def load_language_models(self, model_paths: Dict[str, Dict[str, str]]) -> None:
        """
        Load n-gram profiles from provided file paths for all languages.
        
        Args:
            model_paths: Dictionary mapping language names to their n-gram model file paths
        """
        for language in self.languages:
            self.load_single_language_model(language, model_paths[language])
    
    def load_single_language_model(self, language: str, ngram_files: Dict[str, str]) -> None:
        """
        Load n-gram model for a single language.
        
        Args:
            language: Name of the language
            ngram_files: Dictionary mapping n-gram types to file paths
                        Example: {'2gram': 'path/to/file.json', '3gram': 'path/to/file.json'}
        """
        self.rank_profiles[language] = {}
        self.penalty_rank[language] = {}
        
        for file_key, file_path in ngram_files.items():
            if self.is_valid_ngram_file(file_key):
                ngram_size = self.extract_ngram_size(file_key)
                if ngram_size in self.ngram_range:
                    self.load_ngram_file(language, ngram_size, file_path)
    
    def is_valid_ngram_file(self, file_key: str) -> bool:
        """
        Check if the file key represents a valid n-gram file.
        
        Args:
            file_key: Key from the ngram_files dictionary (e.g., '2gram', '3gram')
            
        Returns:
            True if the key ends with 'gram', False otherwise
        """
        return file_key.endswith("gram")
    
    def extract_ngram_size(self, file_key: str) -> int:
        """
        Extract n-gram size from file key.
        
        Args:
            file_key: Key like '2gram', '3gram', etc.
            
        Returns:
            Integer n-gram size (e.g., 2 for '2gram')
        """
        return int(file_key[:-4])  # Remove 'gram' suffix and convert to int
    
    def load_ngram_file(self, language: str, ngram_size: int, file_path: str) -> None:
        """
        Load a single n-gram file and create rank mappings.
        
        Args:
            language: Language name
            ngram_size: Size of n-grams (e.g., 2 for bigrams)
            file_path: Path to the n-gram frequency file
        """
        # Load n-gram frequencies from JSON file
        frequency_map = self.read_ngram_frequencies(file_path)
        
        # Sort n-grams by frequency and create rank mapping in descending order
        rank_map = self.create_rank_mapping(frequency_map)
        
        # Store rank mapping and penalty rank for this language and n-gram size
        self.rank_profiles[language][ngram_size] = rank_map
        # Penalty-Rank: One rank higher than the highest n-gram rank
        self.penalty_rank[language][ngram_size] = len(rank_map) + 1
    
    def read_ngram_frequencies(self, file_path: str) -> Dict[str, int]:
        """
        Read n-gram frequencies from JSON file.
        
        Args:
            file_path: Path to the JSON file containing n-gram frequencies
            
        Returns:
            Dictionary mapping n-grams to their frequencies
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        
    #--- Out-of-Place Distance Logic ---
    
    def create_rank_mapping(self, frequency_map: Dict[str, int]) -> Dict[str, int]:
        """
        Create rank mapping from frequency data.
        
        Args:
            frequency_map: Dictionary mapping n-grams to their frequencies
            
        Returns:
            Dictionary mapping n-grams to their ranks (1 = most frequent)
            Example: {'the': 1, 'and': 2, 'to': 3, ...}
        """
        sorted_ngrams = sorted(frequency_map.items(), key=lambda item: item[1], reverse=True)
        return detection_helper.create_rank_map(sorted_ngrams)
    
    def out_of_place_distance(self, word: str, language: str) -> float:
        """
        Compute the Out-of-Place-Distance for a single word in a given language.
        
        The out-of-place distance measures how "foreign" a word appears in a given language
        by comparing its n-gram patterns to the expected patterns of that language.
        
        Args:
            word: The word to analyze
            language: The language to compute distance for
            
        Returns:
            Average out-of-place distance across all n-gram sizes
            Lower values indicate better fit to the language
        """
        total_distance = 0.0
        ngram_type_count = 0
        
        # Calculate distance for each n-gram size
        for ngram_size in self.ngram_range:
            distance = self.compute_ngram_distance(word, language, ngram_size)
            if distance is not None:
                total_distance += distance
                ngram_type_count += 1
        
        # Average over all n-gram types (if multiple n are used)
        return total_distance / ngram_type_count if ngram_type_count > 0 else float('inf')
    
    def compute_ngram_distance(self, word: str, language: str, ngram_size: int) -> Optional[float]:
        """
        Compute distance for a specific n-gram size.
        
        Args:
            word: The word to analyze
            language: The language to compute distance for
            ngram_size: Size of n-grams to use
            
        Returns:
            Normalized distance for this n-gram size, or None if no n-grams available
        """
        # Generate n-grams for the word
        ngrams = compute_n_gram([word], ngram_size)
        ngrams_array = np.array(ngrams)
        
        # Skip if no n-grams were generated
        if ngrams_array.size == 0:
            return None
        
        # Get rank mapping and penalty rank for this language and n-gram size
        rank_mapping = self.rank_profiles[language].get(ngram_size, {})
        """
        Example: ngram_size=2
        rank_mapping = {'al': 1, '_ha': 2, ...}
        """
        penalty_rank = self.penalty_rank[language].get(ngram_size, 1000)
        
        # Get ranks for all n-grams and compute total distance
        ngram_ranks = self.get_ngram_ranks(ngrams, rank_mapping, penalty_rank)
        total_distance = np.sum(ngram_ranks)
        
        # Normalized by the number of n-grams
        return total_distance / ngrams_array.size
    
    def get_ngram_ranks(self, ngrams: List[str], rank_mapping: Dict[str, int], penalty_rank: int) -> np.ndarray:
        """
        Get ranks for a list of n-grams using vectorized operations.
        
        Args:
            ngrams: List of n-grams
            rank_mapping: Dictionary mapping n-grams to ranks
            penalty_rank: Rank to assign to unknown n-grams
            
        Returns:
            Array of ranks for each n-gram
        """
        ngrams_array = np.array(ngrams)
        # Use vectorized function to get ranks efficiently
        rank_function = np.vectorize(lambda ngram: rank_mapping.get(ngram, penalty_rank))
        return rank_function(ngrams_array)
    
    def detect_word(self, word: str) -> Dict[str, Union[str, float, None]]:
        """
        Detect the language for a single (padded) word using Out-of-Place distance.
        
        Args:
            word: The input word to analyze
            
        Returns:
            Dictionary with detected language and confidence score
            Example successful detection:
            {
                'word': 'Hello',
                'language': 'English',
                'confidence': 0.95
            }
            Example ambiguous detection:
            {
                'word': 'Hello',
                'language': 'Ambiguous',
                'confidence': None
            }
        """
        # Compute distances to all languages
        language_distances = self.compute_all_language_distances(word)
        sorted_languages = self.sort_languages_by_distance(language_distances)
        
        # Smallest distance = best language
        best_language, best_score = sorted_languages[0]
        # Example: best_language = 'English', best_score = 0.1234 
        # out of e.g. [('English', 0.1234), ('German', 0.2345), ...]
        second_language, second_score = detection_helper.compute_second_best_language(sorted_languages)
        
        # Ambiguous if the scores are too similar or too large
        if self.is_detection_ambiguous(best_score, second_score):
            return self.create_ambiguous_result(word)
        
        # Calculate confidence based on score difference
        confidence = detection_helper.compute_confidence(best_score, second_score)
        return self.create_detection_result(word, best_language, confidence)
    
    def compute_all_language_distances(self, word: str) -> Dict[str, float]:
        """
        Compute out-of-place distances for all languages.
        
        Args:
            word: The word to analyze
            
        Returns:
            Dictionary mapping language names to their distances
        """
        distances = {}
        for language in self.languages:
            distances[language] = self.out_of_place_distance(word, language)
        return distances
    
    def sort_languages_by_distance(self, distances: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Sort languages by their distance scores (ascending).
        
        Args:
            distances: Dictionary mapping languages to distances
            
        Returns:
            List of (language, distance) tuples sorted by distance
        """
        return sorted(distances.items(), key=lambda item: item[1])
    
    def is_detection_ambiguous(self, best_score: float, second_score: float) -> bool:
        """
        Check if detection result is ambiguous based on scores.
        
        Args:
            best_score: Distance score of the best language
            second_score: Distance score of the second-best language
            
        Returns:
            True if the detection should be considered ambiguous
        """
        return detection_helper.is_ambiguous(best_score, second_score)
    
    def create_ambiguous_result(self, word: str) -> Dict[str, Union[str, None]]:
        """
        Create result dictionary for ambiguous detection.
        
        Args:
            word: The word that was analyzed
            
        Returns:
            Dictionary with ambiguous result format
        """
        return {
            'word': word,
            'language': self.AMBIGUOUS_LANGUAGE,
            'confidence': None
        }
    
    def create_detection_result(self, word: str, language: str, confidence: float) -> Dict[str, Union[str, float]]:
        """
        Create result dictionary for successful detection.
        
        Args:
            word: The word that was analyzed
            language: The detected language
            confidence: The confidence score
            
        Returns:
            Dictionary with detection result
        """
        return {
            'word': word,
            'language': language,
            'confidence': round(confidence, 2)
        }
    
    #--- Smoothing ---
    
    def apply_context_smoothing(self, results: List[Dict], window: int) -> List[Dict]:
        """
        Smooth ambiguous language detections using surrounding context.
        
        Context smoothing works by looking at surrounding words to help determine
        the language of ambiguous words. If neighboring words strongly suggest
        a particular language, that language is assigned to the ambiguous word.
        
        Args:
            results: List of word language detection results
                    Example:
                    [
                        {'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                        {'word': 'Welt', 'language': 'Ambiguous', 'confidence': None},
                        {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.85},
                        ...
                    ]
            window: Number of surrounding words to consider for context
                   
        Returns:
            List of results with smoothed language assignments for ambiguous words
        """
        # If there are no words that could benefit from context, return results as is
        if len(results) <= 1:
            return results
        
        # Create a copy of results to avoid modifying the original list
        smoothed_results = results.copy()
        
        # Iterate over every word and check for Ambiguous detections
        # If a word is Ambiguous, look at the context to determine the best language
        for current_index in range(len(results)):
            if self.is_word_ambiguous(results[current_index]):
                # Extract surrounding words as context
                context_words = self.extract_context_window(results, current_index, window)
                
                # Calculate language votes based on context
                language_votes = confidence.calculate_language_votes(context_words, self.is_valid_vote)
                print(language_votes)
                
                # Apply smoothing if confidence threshold is met
                if confidence.should_apply_smoothing(language_votes, self.MIN_CONFIDENCE_THRESHOLD):
                    best_language = self.get_best_voted_language(language_votes)
                    smoothed_confidence = confidence.calculate_smoothed_confidence(language_votes[best_language], self.BASE_SMOOTHED_CONFIDENCE, 
                                                                                          self.MAX_SMOOTHED_CONFIDENCE, self.CONFIDENCE_SCALING_FACTOR)
                    # Update the word's detection result
                    self.update_word_detection(smoothed_results[current_index], best_language, smoothed_confidence)
        return smoothed_results
    
    def is_word_ambiguous(self, word_result: Dict) -> bool:
        """
        Check if a word detection result is ambiguous.
        
        Args:
            word_result: Dictionary containing word detection result
        Returns:
            True if the word was detected as ambiguous
        """
        return word_result['language'] == self.AMBIGUOUS_LANGUAGE
    
    def extract_context_window(self, results: List[Dict], current_index: int, window: int) -> List[Dict]:
        """
        Extract context words around the current word position.
        
        Args:
            results: Full list of detection results
            current_index: Index of the current word being processed
            window: Number of words before and after to include (i = current index, window = number of words to consider before and after)
            
        Returns:
            List of context words (excluding the current word)
            
            Example:
            If results = [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                        {'word': 'Welt', 'language': 'Ambiguous', 'confidence': None},
                        {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.85}]
            and window = 1, then context = [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                        {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.85}]
        """
        # Get context window boundaries
        start_index = max(0, current_index - window)
        end_index = min(len(results), current_index + window + 1)
        
        # Return context excluding the current word
        before_current = results[start_index:current_index]
        after_current = results[current_index + 1:end_index]
        
        return before_current + after_current
    
    
    
    def is_valid_vote(self, word_result: Dict) -> bool:
        """
        Check if a word result can contribute to language voting.
        
        Args:
            word_result: Dictionary containing word detection result
            
        Returns:
            True if the word can be used for voting (not ambiguous/unknown and has confidence)
        """
        language = word_result['language']
        confidence = word_result['confidence']
        
        invalid_languages = {self.AMBIGUOUS_LANGUAGE, self.UNKNOWN_LANGUAGE}
        return language not in invalid_languages and confidence is not None
    
    
    def get_best_voted_language(self, language_votes: Dict[str, float]) -> str:
        """
        Get the language with the highest confidence score.
        
        Args:
            language_votes: Dictionary mapping languages to vote scores
            
        Returns:
            Language name with the highest vote score
        """
        return max(language_votes, key=language_votes.get)

    
    def update_word_detection(self, word_result: Dict, language: str, confidence: float) -> None:
        """
        Update a word detection result with new language and confidence.
        
        Args:
            word_result: Dictionary to update (modified in place)
            language: New language assignment
            confidence: New confidence score
        """
        word_result['language'] = language
        word_result['confidence'] = confidence
    
    
    def detect_text_languages(self, text: str, context_window: int = DEFAULT_CONTEXT_WINDOW) -> List[Dict]:
        """
        Detect the language for each word in a text using the provided detection function.
        
        Args:
            text: The input text to analyze
            context_window: Number of words to consider for context smoothing
            
        Returns:
            List of dictionaries with detected languages and confidence scores for each word
            Example:
            [
                {'word': 'hello', 'language': 'English', 'confidence': 0.95},
                {'word': 'welt', 'language': 'German', 'confidence': 0.85},
                ...
            ]
        """
        # Tokenize the input text with padding
        tokens = tokenizeWithPadding(text)
        
        # Detect language for each word individually
        detection_results = detection_helper.detect_language_for_each_word(self.detect_word, tokens)
        
        # Apply context smoothing to resolve ambiguous detections
        return self.apply_context_smoothing(detection_results, context_window)

    
    