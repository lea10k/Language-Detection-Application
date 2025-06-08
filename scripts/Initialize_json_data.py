import n_gram_frequencies
import n_gram
import tokenization

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/english'

tokenized_text = tokenization.tokenizeWithPadding(n_gram_frequencies.GetWholeText(corpus_root))

# Training data is already injected into the file
"""n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 2), 'Language-Detection-Application/json_data/german/2grams.json')

n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 3), 'Language-Detection-Application/json_data/german/3grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 4), 'Language-Detection-Application/json_data/german/4grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 5), 'Language-Detection-Application/json_data/german/5grams.json')

n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequenciesForUnigrams(tokenized_text), 'Language-Detection-Application/json_data/german/1grams.json')"""

#n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeRelativeFrequencies(tokenized_text, 1), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/1grams.json')

n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeFrequencies(tokenized_text, 1), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/1grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeFrequencies(tokenized_text, 2), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/2grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeFrequencies(tokenized_text, 3), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/3grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeFrequencies(tokenized_text, 4), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/4grams.json')
n_gram_frequencies.SaveAsJSON(n_gram_frequencies.ComputeFrequencies(tokenized_text, 5), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/5grams.json')