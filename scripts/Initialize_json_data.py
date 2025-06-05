import n_gram_frequencies
import tokenization

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/italian'

tokenized_text = tokenization.tokenizeWithPadding(n_gram_frequencies.GetWholeText(corpus_root))

# Training data is already injected into the file
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 2), 'Language-Detection-Application/json_data/italian/2grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 5), 'Language-Detection-Application/json_data/italian/5grams.json')