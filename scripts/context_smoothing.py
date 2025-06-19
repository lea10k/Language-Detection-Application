import confidence

class ContextSmoother:
    """
    Smooths ambiguous detections using surrounding context.
    """
    def apply_context_smoothing(self, results, window):
        if len(results) <= 1:
            return results
        smoothed = results.copy()
        for i, res in enumerate(results):
            if self.is_word_ambiguous(res):
                ctx = self.extract_context_window(results, i, window)
                votes = confidence.calculate_language_votes(ctx, self.is_valid_vote)
                if confidence.should_apply_smoothing(votes, self.MIN_CONFIDENCE_THRESHOLD):
                    lang = self.get_best_voted_language(votes)
                    conf = confidence.calculate_smoothed_confidence(
                        votes[lang], self.BASE_SMOOTHED_CONFIDENCE,
                        self.MAX_SMOOTHED_CONFIDENCE,
                        self.CONFIDENCE_SCALING_FACTOR)
                    self.update_word_detection(smoothed[i], lang, conf)
        return smoothed

    def is_word_ambiguous(self, word_result):
        return word_result['language'] == self.AMBIGUOUS_LANGUAGE

    def extract_context_window(self, results, idx, window):
        start = max(0, idx - window)
        end = min(len(results), idx + window + 1)
        return results[start:idx] + results[idx+1:end]

    def is_valid_vote(self, word_result):
        lang = word_result['language']
        conf = word_result['confidence']
        return lang not in {self.AMBIGUOUS_LANGUAGE, self.UNKNOWN_LANGUAGE} and conf is not None

    def get_best_voted_language(self, votes):
        return max(votes, key=votes.get)

    def update_word_detection(self, word_result, language, confidence):
        word_result['language'] = language
        word_result['confidence'] = confidence
