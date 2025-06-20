import confidence

class ContextSmoother:
    """
    Smooths ambiguous detections using surrounding context.
    """
    def apply_context_smoothing(self, results, window) -> list:
        """
        Apply context smoothing to ambiguous word detections.
        args:
            results (list): List of word detection results, each a dict with 'language' and 'confidence'.
            window (int): Number of surrounding words to consider for smoothing.
        returns:
            list: Smoothed results with updated language and confidence for ambiguous words.
        Example:
            If results = [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
                          {'word': 'Ciao', 'language': 'Ambiguous', 'confidence': None},
                          {'word': 'Mamma', 'language': 'Italian', 'confidence': 0.85}],
            and window = 1, the output might be:
            [{'word': 'Hello', 'language': 'English', 'confidence': 0.95},
             {'word': 'Ciao', 'language': 'Italian', 'confidence': 0.80},
             {'word': 'Mamma', 'language': 'Italian', 'confidence': 0.85}]
        """
        if len(results) <= 1:
            return results
        
        smoothed = results.copy()
        
        for i, res in enumerate(results):
            if self.is_word_ambiguous(res):
                ctx = self.extract_context_window(results, i, window)
                votes = confidence.calculate_language_votes(ctx, self.is_valid_vote)
                if self.should_apply_smoothing(votes):
                    lang = self.get_best_voted_language(votes)
                    conf = confidence.calculate_smoothed_confidence(
                        votes[lang], self.BASE_SMOOTHED_CONFIDENCE,
                        self.MAX_SMOOTHED_CONFIDENCE,
                        self.CONFIDENCE_SCALING_FACTOR)
                    self.update_word_detection(smoothed[i], lang, conf)
        return smoothed

    def is_word_ambiguous(self, word_result) -> bool:
        return word_result['language'] == self.AMBIGUOUS_LANGUAGE

    def extract_context_window(self, results, idx, window) -> list:
        """
        Extract a context window of results around the given index.
        Args:
            results (list): List of word detection results.
            idx (int): Index of the word to extract context for.
            window (int): Number of surrounding words to include in the context.
        Returns:
            list: Context words excluding the word at idx.
        """
        start = max(0, idx - window)
        end = min(len(results), idx + window + 1)
        return results[start:idx] + results[idx+1:end]

    def is_valid_vote(self, word_result) -> bool:
        """
        Check if a word detection result is a valid vote. Means it is not ambiguous or unknown.
        """
        lang = word_result['language']
        conf = word_result['confidence']
        return lang not in {self.AMBIGUOUS_LANGUAGE, self.UNKNOWN_LANGUAGE} and conf is not None

    def should_apply_smoothing(self, language_votes: dict) -> bool:
        """
        Decide whether to apply smoothing based on language vote confidence.

        Smoothing is applied if there is at least one language vote and
        the highest confidence among the votes is greater than or equal to
        MIN_CONFIDENCE_THRESHOLD.

        Args:
            language_votes (dict): Mapping of language to confidence score.

        Returns:
            bool: True if smoothing should be applied, False otherwise.
        """
        if not language_votes:
            return False
        highest_confidence = max(language_votes.values())
        return highest_confidence >= self.MIN_CONFIDENCE_THRESHOLD

    def get_best_voted_language(self, votes) -> str:
        """
        Get the language with the highest vote.
        """
        return max(votes, key=votes.get)

    def update_word_detection(self, word_result, language, confidence):
        word_result['language'] = language
        word_result['confidence'] = confidence
