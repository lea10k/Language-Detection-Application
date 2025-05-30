import tokenization
import n_gram
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/german'
corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
corpus_files = corpus.fileids()

"""for file_id in corpus_files:
    text = corpus.raw(file_id)
    # Tokenize and create n-grams"""
   

tokens = tokenization.tokenizeWithPadding(corpus.raw('1.txt'))
three_grams = n_gram.compute_n_gram(tokens, 3)
# Compute frequency distribution
freq_dist = FreqDist(three_grams)
print(f"Frequency distribution for {'1.txt'}:\n{freq_dist.most_common(20)}\n")

tokens = tokenization.tokenizeWithPadding(corpus.raw('1.txt'))
three_grams = n_gram.compute_n_gram(tokens, 4)
# Compute frequency distribution
freq_dist = FreqDist(three_grams)
print(f"Frequency distribution for {'1.txt'}:\n{freq_dist.most_common(20)}\n")