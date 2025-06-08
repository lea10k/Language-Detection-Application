import json
from collections import defaultdict
from tokenization import tokenizeWithPadding
from n_gram import compute_n_gram

class WordLevelLanguageDetector:
    """
    Klar strukturierter Language Detector auf Basis von Out-of-Place-Distanz.
    Die Profile bestehen aus Ranglisten von n-Grammen.
    """
    def __init__(self, model_paths: dict[str, dict[str, str]], ngram_range=(2, 3, 4, 5)):
        self.languages = list(model_paths.keys())
        self.ngram_range = ngram_range  
        self.rank_profiles = {}  # {lang: {n: {gram: rank}}}
        self.penalty_rank = {}   # {lang: {n: K}}
        
        # Load n-gram profiles from provided paths
        for lang, ngram_files in model_paths.items():
            self.rank_profiles[lang] = {}
            self.penalty_rank[lang] = {}
            for key, path in ngram_files.items():
                if not key.endswith("gram"):
                    continue
                n = int(key[:-4])
                if n not in self.ngram_range:
                    continue
                # Load n-gram frequencies from JSON file
                with open(path, 'r', encoding='utf-8') as f:
                    freq_map = json.load(f)
                # Sort n-grams by frequency and create rank mapping in descending order
                sorted_grams = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
                rank_map = {gram: rank for rank, (gram, _) in enumerate(sorted_grams, start=1)}
                self.rank_profiles[lang][n] = rank_map
                # Penalty-Rank: One rank higher than the highest n-gram rank
                self.penalty_rank[lang][n] = len(rank_map) + 1

    def _out_of_place_distance(self, word: str, lang: str) -> float:
        """Computes the Out-of-Place-Distance for a single word in a given language."""
        total_dist = 0.0
        ngram_counts = 0
        for n in self.ngram_range:
            grams = compute_n_gram([word], n)
            ranks = self.rank_profiles[lang].get(n, {})
            K = self.penalty_rank[lang].get(n, 1000)
            # For each n-gram: rank in profile or penalty
            dist = sum(ranks.get(g, K) for g in grams)
            # Normalized by the number of n-grams
            total_grams = len(grams)
            if total_grams > 0:
                total_dist += dist / total_grams
                ngram_counts += 1
        # Average over all n-gram types (if multiple n are used)
        return total_dist / ngram_counts if ngram_counts else float('inf')

    def detect_word(self, word: str, ambiguity_margin=0.25) -> dict:
        """Detects the language for a single (padded) word using Out-of-Place distance."""
        distances = {lang: self._out_of_place_distance(word, lang) for lang in self.languages}
        # Smallest distance = best language
        sorted_langs = sorted(distances.items(), key=lambda kv: kv[1])
        best_lang, best_score = sorted_langs[0]
        second_lang, second_score = sorted_langs[1] if len(sorted_langs) > 1 else (None, float('inf'))
        
        # Ambiguous if the scores are too similar or too large
        if second_score < float('inf') and (second_score - best_score) / best_score < ambiguity_margin:
            return {'word': word, 'language': 'ambiguous', 'confidence': None}
        
        confidence = min(0.95, max(0.1, (second_score - best_score) / (second_score if second_score > 0 else 1)))
        return {'word': word, 'language': best_lang, 'confidence': round(confidence, 2)}

    def detect_text_languages(self, text: str, context_window=3) -> list[dict]:
        tokens = tokenizeWithPadding(text)
        results = [self.detect_word(token) for token in tokens]
        return self._apply_context_smoothing(results, context_window)

    def _apply_context_smoothing(self, results: list[dict], window: int) -> list[dict]:
        """Smooth ambiguity using context."""
        if len(results) <= 1:
            return results
        smoothed = results.copy()
        for i, result in enumerate(results):
            if result['language'] == 'ambiguous':
                # Get context window
                start = max(0, i - window)
                end = min(len(results), i + window + 1)
                context = results[start:i] + results[i+1:end]
                votes = defaultdict(float)
                for ctx in context:
                    if ctx['language'] not in ['ambiguous', 'unknown'] and ctx['confidence']:
                        votes[ctx['language']] += ctx['confidence']
                if votes:
                    best_lang = max(votes, key=votes.get)
                    if votes[best_lang] >= 1.0:
                        smoothed[i]['language'] = best_lang
                        smoothed[i]['confidence'] = min(0.8, 0.3 + votes[best_lang] * 0.1)
        return smoothed

    def get_language_summary(self, results: list[dict]) -> dict:
        lang_counts = defaultdict(int)
        total_confidence = defaultdict(float)
        for result in results:
            lang = result['language']
            if lang not in ['ambiguous', 'unknown']:
                lang_counts[lang] += 1
                total_confidence[lang] += result['confidence'] or 0
        summary = {}
        total_words = sum(lang_counts.values())
        for lang, count in lang_counts.items():
            avg_confidence = total_confidence[lang] / count if count > 0 else 0.0
            summary[lang] = {
                'words': count,
                'percentage': round((count / total_words) * 100, 1) if total_words > 0 else 0.0,
                'avg_confidence': round(avg_confidence, 2)
            }
        return summary
