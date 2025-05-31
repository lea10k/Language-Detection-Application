import n_gram_frequencies
import tokenization

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/german'

tokenized_text = tokenization.tokenizeWithPadding(n_gram_frequencies.GetWholeText(corpus_root))

# Training data is already injected into the file
# SaveAsJSON(ComputeRelativeFrequencies(tokenized_text, 3), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/3grams.json')