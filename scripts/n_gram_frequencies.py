import tokenization
import n_gram
import json
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

"""
Get the whole text from the data set

:param corpus_files: List of file IDs in the corpus
:param corpus: The corpus to read from (default is a PlaintextCorpusReader)
"""
def GetWholeText(corpus_files, corpus=PlaintextCorpusReader('corpus', '.*')):
    text = ''
    for file_id in corpus_files:
        text += corpus.raw(file_id)
    return text

"""
Function to extract the second element (frequency count) from a tuple

:param item: A tuple where the second element is the frequency count
:returns: The second element of the tuple
"""
def keyFunction(item):
    return item[1]

"""
Sort the frequency distribution in descending order based on frequency counts

:param freq_dist: Frequency distribution dictionary
:returns: Sorted frequency distribution dictionary
"""
def SortFrequencies(freq_dist):
    sorted_freq = sorted(freq_dist.items(), key=keyFunction, reverse=True)
    dic_freq_dist = dict(sorted_freq)
    return dic_freq_dist

"""
Compute the frequencies of n-grams in the tokenized text

:param tokenized_text: List of tokenized words
:param n: The n-gram size
"""
def ComputeFrequencies(tokenized_text, n):
    n_grams = n_gram.compute_n_gram(tokenized_text, n)
    # Compute frequency distribution
    freq_dist = FreqDist(n_grams)
    return SortFrequencies(freq_dist)

"""
Compute the relative frequencies of n-grams in the tokenized text

:param tokenized_text: List of tokenized words
:param n: The n-gram size
"""
def ComputeRelativeFrequencies(tokenized_text, n):
    freq_dist = ComputeFrequencies(tokenized_text, n)
    total_count = sum(freq_dist.values())
    relative_freq_dist = {k: v / total_count for k, v in freq_dist.items()}
    return relative_freq_dist

"""
Save the computed frequencies as a JSON file
:param data: The data to save (frequency distribution)
:param filename: The name of the file to save the data to
"""
def SaveAsJSON(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# DEV SET EINBINDEN
tokenized_text = tokenization.tokenizeWithPadding(GetWholeText(corpus_files))

# Check if the sum of relative frequencies is 1 to ensure correctness of the computation
rel_freqs = ComputeRelativeFrequencies(tokenized_text, 3)
print(sum(rel_freqs.values()))

