import json
import detection_helper

class LanguageModelLoader:
    """
    Handles loading of n-gram frequency profiles and creation of rank mappings.
    """
    def load_language_models(self, model_paths):
        """
        Load n-gram profiles for all languages.
        """
        self.rank_profiles = {}
        self.penalty_rank = {}
        for language, paths in model_paths.items():
            self.rank_profiles[language] = {}
            self.penalty_rank[language] = {}
            self.load_single_language_model(language, paths)

    def load_single_language_model(self, language, ngram_files):
        for file_key, file_path in ngram_files.items():
            if self.is_valid_ngram_file(file_key):
                n = self.extract_ngram_size(file_key)
                if n in self.ngram_range:
                    freq_map = self.read_ngram_frequencies(file_path)
                    rank_map = self.create_rank_mapping(freq_map)
                    self.rank_profiles[language][n] = rank_map
                    self.penalty_rank[language][n] = len(rank_map) + 1

    def is_valid_ngram_file(self, file_key):
        return file_key.endswith("gram")

    def extract_ngram_size(self, file_key):
        return int(file_key[:-4])

    def read_ngram_frequencies(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_rank_mapping(self, frequency_map):
        sorted_ngrams = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        return detection_helper.create_rank_map(sorted_ngrams)

