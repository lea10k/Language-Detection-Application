import numpy as np
from n_gram_computation import compute_n_gram

class LanguageDistanceCalculator:
    """
    Computes mean rank distances between words and language profiles.
    """
    
    def mean_rank_distance(self, word, language) -> float:
        """
        Compute the mean rank distance for a word in a specific language across all n-gram profiles.
        Args:
            word (str): The word to compute the distance for.
            language (str): The language to compute the distance against.
        Returns:
            float: The mean rank distance averaged over all n-gram sizes.
            Example:
            If word = "Hello", language = "English", the output might be:
            2.5
        """
        total_distance = 0.0
        number_of_ngram_profiles = 0
        for n in self.ngram_range:
            dist = self.compute_mean_ngram_distance(word, language, n)
            if dist is not None:
                total_distance += dist
                number_of_ngram_profiles += 1
        distance = total_distance / number_of_ngram_profiles if number_of_ngram_profiles > 0 else float('inf')
        #print(f"Mean rank distance across all n-gram profiles for word '{word}' in language '{language}': {distance}")
        return distance
    

    def compute_mean_ngram_distance(self, word, language, n) -> float:
        """
        Compute the mean rank distance for a word in a specific language and n-gram size.
        Args:
            word (str): The word to compute the distance for.
            language (str): The language to compute the distance against.
            n (int): The size of the n-grams to use for the distance computation.
        Returns:
            float: The mean rank distance for the given n-gram size.
            Example:
            If word = "Hello", language = "English", n = 2, the output might be:
            3.5
        """
        ngrams_of_word = compute_n_gram([word], n)
        if len(ngrams_of_word) == 0:
            return None
        rank_map = self.rank_profiles.get(language, {}).get(n, {}) #rank_profiles from LanguageModelLoader
        penalty = self.penalty_rank.get(language, {}).get(n, 1000)#penalty_rank from LanguageModelLoader
        ranks = self.get_ngram_ranks(ngrams_of_word, rank_map, penalty)
        distance = np.sum(ranks) / len(ngrams_of_word) #ranks of word n-grams from n-gram profiles 
        #print(f"Computed mean rank distance for word '{word}' in language '{language}' with n={n}: {distance}")
        return distance

    def get_ngram_ranks(self, ngrams_of_word, rank_mapping, penalty_rank) -> np.ndarray:
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
        vec = np.vectorize(lambda gram: rank_mapping.get(gram, penalty_rank)) # search for each n-gram in the rank mapping of a given language, use penalty rank if not found
        #print(f"Rank mapping for n-grams: {rank_mapping}, Penalty rank: {penalty_rank}")
        return vec(ngram_arr)

    def compute_all_language_distances(self, word) -> dict:
        """
        Compute the mean rank distances for a word across all languages.
        Args:
            word (str): The word to compute distances for.
        Returns:
            dict: A dictionary mapping languages to their mean rank distances for the given word.
        """
        #print(f"Computing distances for word: {word}")
        return {lang: self.mean_rank_distance(word, lang) for lang in self.languages}

    def sort_languages_by_distance(self, distances) -> list:
        """
        Sort languages by their mean rank distances.
        Args:
            distances (dict): A dictionary mapping languages to their mean rank distances.
        Returns:
            list: A sorted list of tuples (language, distance) sorted by distance in ascending order.
        """
        sorted_languages = sorted(distances.items(), key=lambda x: x[1])
        #print(f"Sorted languages by distance for word: {sorted_languages}")
        return sorted_languages
