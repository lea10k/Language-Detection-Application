import json
import detection_helper

class LanguageModelLoader:
    """
    Handles loading of n-gram frequency profiles and creation of rank mappings.
    """
    def load_language_models(self, model_paths):
        """
        Loads n-gram profiles for all specified languages.

        For each language in the `model_paths` dictionary, the corresponding n-gram models are loaded
        and stored in the attributes `self.rank_profiles` and `self.penalty_rank`.
        The method calls the helper method `load_single_language_model` for each language.

        Args:
            model_paths (dict): A dictionary where each key is a language and each value is a dictionary
            containing the paths to the corresponding n-gram models.

        Side Effects:
            Updates the attributes `self.rank_profiles` and `self.penalty_rank` with the loaded profiles.
        """
        self.rank_profiles = {} # Structure when filled: {lang: {n: {gram: rank}}}
        self.penalty_rank = {} # Structure when filled: {lang: {n: penalty_rank}}
        for language, paths in model_paths.items():
            self.rank_profiles[language] = {}
            self.penalty_rank[language] = {}
            # Load n-gram profiles from provided paths
            self.load_single_language_model(language, paths)

    def load_single_language_model(self, language, ngram_files):
        """
        Creates rank profiles and penalty ranks for a single language based on n-gram frequency files.
        Args:
            language: Name of the language
            ngram_files: Dictionary mapping n-gram types to file paths
                         Example: {'2gram': 'path/to/file.json', '3gram': 'path/to/file.json'}
        Side Effects:
            Populates `self.rank_profiles` and `self.penalty_rank` for the specified language
            with n-gram frequencies and their ranks.
        """
        for file_key, file_path in ngram_files.items():
            if self.is_valid_ngram_file(file_key):
                n = self.extract_ngram_size(file_key)
                if n in self.ngram_range:
                    freq_map = self.read_ngram_frequencies(file_path)
                    rank_map = self.create_rank_mapping(freq_map)
                    self.rank_profiles[language][n] = rank_map
                    # Penalty rank is one more than the highest rank in the rank_map
                    self.penalty_rank[language][n] = len(rank_map) + 1

    def is_valid_ngram_file(self, file_key):
        return file_key.endswith("gram")

    def extract_ngram_size(self, file_key):
        return int(file_key[:-4])

    def read_ngram_frequencies(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_rank_mapping(self, frequency_map):
        """
        Create a rank mapping from the frequency map.
        args:
            frequency_map (dict): A dictionary where keys are n-grams and values are their frequencies
        returns:
            rank_map (dict): A dictionary where keys are n-grams and values are their ranks
        Example: {'al': 1, '_ha': 2, ...}
        """
        # Sort the frequency map by frequency in descending order
        sorted_ngrams = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        # (gram, _) is taking only the n-gram, not the frequency. _ ignores the frequency value.
        rank_map = {gram: rank for rank, (gram, _) in enumerate(sorted_ngrams, start=1)}
        return rank_map

# WO WIRD DER PENALTY RANK EINGEFÜGT?