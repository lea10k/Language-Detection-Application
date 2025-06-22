import json
from n_gram_computation import compute_n_gram
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist

def get_whole_text(corpus_root):
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

def key_function(item):
    """
    Function to extract the second element (frequency count) from a tuple

    :param item: A tuple where the second element is the frequency count
    :returns: The second element of the tuple
    """
    return item[1]

def sort_frequencies(freq_dist):
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

def compute_frequencies(tokenized_text, n):
    """
    Compute the frequencies of n-grams in the tokenized text

    :param tokenized_text: List of tokenized words
    :param n: The n-gram size
    """
    n_grams = compute_n_gram(tokenized_text, n)
    # Compute frequency distribution
    freq_dist = FreqDist(n_grams)
    return SortFrequencies(freq_dist)

def compute_frequencies_for_unigrams(tokenized_text):
    """
    Compute the frequencies of unigrams in the tokenized text

    :param tokenized_text: List of tokenized words
    """
    # Compute frequency distribution for unigrams
    freq_dist = FreqDist(tokenized_text)
    return SortFrequencies(freq_dist)

def save_as_json(data, filename):
    """
    Save the computed frequencies as a JSON file
    :param data: The data to save (frequency distribution)
    :param filename: The name of the file to save the data to
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)


