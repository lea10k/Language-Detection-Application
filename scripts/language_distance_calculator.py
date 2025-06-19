import numpy as np
from n_gram_computation import compute_n_gram

class LanguageDistanceCalculator:
    """
    Computes out-of-place distances between words and language profiles.
    """
    def out_of_place_distance(self, word, language):
        total = 0.0
        count = 0
        for n in self.ngram_range:
            dist = self.compute_ngram_distance(word, language, n)
            if dist is not None:
                total += dist
                count += 1
        return total / count if count > 0 else float('inf')

    def compute_ngram_distance(self, word, language, n):
        ngrams = compute_n_gram([word], n)
        if len(ngrams) == 0:
            return None
        rank_map = self.rank_profiles.get(language, {}).get(n, {})
        penalty = self.penalty_rank.get(language, {}).get(n, 1000)
        ranks = self.get_ngram_ranks(ngrams, rank_map, penalty)
        return np.sum(ranks) / len(ngrams)

    def get_ngram_ranks(self, ngrams, rank_mapping, penalty_rank):
        arr = np.array(ngrams)
        vec = np.vectorize(lambda g: rank_mapping.get(g, penalty_rank))
        return vec(arr)

    def compute_all_language_distances(self, word):
        return {lang: self.out_of_place_distance(word, lang) for lang in self.languages}

    def sort_languages_by_distance(self, distances):
        return sorted(distances.items(), key=lambda x: x[1])
