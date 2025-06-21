import numpy as np
from n_gram_computation import compute_n_gram

class LanguageDistanceCalculator:
    """
    Computes mean rank distances between words and language profiles.
    """
    
    def mean_rank_distance(self, word, language):
        """
        Compute the mean rank distance for a word in a specific language across all n-gram profiles.
        Args:
            word (str): The word to compute the distance for.
            language (str): The language to compute the distance against.
        Returns:
            float: The mean rank distance averaged over all n-gram sizes.
        """
        total = 0.0
        count = 0
        for n in self.ngram_range:
            dist = self.compute_mean_ngram_distance(word, language, n)
            if dist is not None:
                total += dist
                count += 1
        return total / count if count > 0 else float('inf')
    

    def compute_mean_ngram_distance(self, word, language, n):
        """
        Compute the mean rank distance for a word in a specific language and n-gram size.
        Args:
            word (str): The word to compute the distance for.
            language (str): The language to compute the distance against.
            n (int): The size of the n-grams to use for the distance computation.
        Returns:
            float: The mean rank distance for the given n-gram size.
        """
        ngrams_of_word = compute_n_gram([word], n)
        if len(ngrams_of_word) == 0:
            return None
        rank_map = self.rank_profiles.get(language, {}).get(n, {})
        penalty = self.penalty_rank.get(language, {}).get(n, 1000)
        ranks = self.get_ngram_ranks(ngrams_of_word, rank_map, penalty)
        return np.sum(ranks) / len(ngrams_of_word)

    def get_ngram_ranks(self, ngrams_of_word, rank_mapping, penalty_rank):
        """
        Get the ranks for the n-grams of a word based on the rank mapping and penalty rank.
        args:
            ngrams_of_word (list): List of n-grams extracted from the word.
            rank_mapping (dict): A dictionary mapping n-grams to their ranks.
            penalty_rank (int): The rank to use for n-grams not found in the mapping
        returns:
            np.ndarray: An array of ranks corresponding to the n-grams of the word.
        """
        ngram_arr = np.array(ngrams_of_word)
        vec = np.vectorize(lambda gram: rank_mapping.get(gram, penalty_rank)) # search for each n-gram in the rank mapping, use penalty rank if not found
        #print(f"Rank mapping for n-grams: {rank_mapping}, Penalty rank: {penalty_rank}")
        return vec(ngram_arr)

    def compute_all_language_distances(self, word):
        #print(f"Computing distances for word: {word}")
        return {lang: self.mean_rank_distance(word, lang) for lang in self.languages}

    def sort_languages_by_distance(self, distances):
        #print(f"Distances before sorting: {distances}")
        return sorted(distances.items(), key=lambda x: x[1])
