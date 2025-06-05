from n_gram import compute_n_gram
from tokenization import tokenizeWithPadding
import json
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

def GetWholeText(corpus_root):
    """
    Get the whole text from the data set

    :param corpus_files: List of file IDs in the corpus
    :param corpus: The corpus to read from (default is a PlaintextCorpusReader)
    """
    corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
    corpus_files = corpus.fileids()
    text = ''
    
    for file_id in corpus_files:
        text += corpus.raw(file_id)
    return text

def keyFunction(item):
    """
    Function to extract the second element (frequency count) from a tuple

    :param item: A tuple where the second element is the frequency count
    :returns: The second element of the tuple
    """
    return item[1]

def SortFrequencies(freq_dist):
    """
    Sort the frequency distribution in descending order based on frequency counts
    
    :param freq_dist: Frequency distribution (dictionary or list of tuples)
    :returns: Sorted frequency distribution dictionary
    """
    # Check if input is a dictionary
    if isinstance(freq_dist, dict):
        sorted_freq = sorted(freq_dist.items(), key=keyFunction, reverse=True)
    # Check if input is a list of tuples
    elif isinstance(freq_dist, list):
        sorted_freq = sorted(freq_dist, key=keyFunction, reverse=True)
    else:
        raise TypeError("Input must be either a dictionary or a list of tuples")
    
    dic_freq_dist = dict(sorted_freq)
    return dic_freq_dist

def ComputeFrequencies(tokenized_text, n):
    """
    Compute the frequencies of n-grams in the tokenized text

    :param tokenized_text: List of tokenized words
    :param n: The n-gram size
    """
    n_grams = compute_n_gram(tokenized_text, n)
    # Compute frequency distribution
    freq_dist = FreqDist(n_grams)
    return SortFrequencies(freq_dist)

def ComputeRelativeFrequencies(tokenized_text, n):
    """
    Compute the relative frequencies of n-grams in the tokenized text

    :param tokenized_text: List of tokenized words
    :param n: The n-gram size
    """
    freq_dist = ComputeFrequencies(tokenized_text, n)
    total_count = sum(freq_dist.values())
    relative_freq_dist = {k: round(v / total_count, 10) for k, v in freq_dist.items()}
    return relative_freq_dist

#print(ComputeRelativeFrequencies(tokenization.tokenizeWithPadding(GetWholeText('/home/lea_k/language_detection_project/Language-Detection-Application/data/development/german')), 3))

def SaveAsJSON(data, filename):
    """
    Save the computed frequencies as a JSON file
    :param data: The data to save (frequency distribution)
    :param filename: The name of the file to save the data to
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# Test the functions with a sample corpus
corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/development/german'
tokenized_text = tokenizeWithPadding(GetWholeText(corpus_root))

# Check if the sum of relative frequencies is 1 to ensure correctness of the computation
rel_freqs = ComputeRelativeFrequencies(tokenized_text, 3)
print(sum(rel_freqs.values()))

