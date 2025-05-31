import n_gram_frequencies
import tokenization
from nltk.corpus import PlaintextCorpusReader

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/german'
corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
corpus_files = corpus.fileids()

tokenized_text = tokenization.tokenizeWithPadding(n_gram_frequencies.GetWholeText(corpus_files))

# Training data is already injected into the file
# SaveAsJSON(ComputeRelativeFrequencies(tokenized_text, 3), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/3grams.json')