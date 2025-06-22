import initialization.extract_data_helper as extract_data_helper
import tokenization

corpus_root = '/home/lea_k/language_detection_project/Language-Detection-Application/data/training/english'

tokenized_text = tokenization.tokenize_with_padding(extract_data_helper.get_whole_text(corpus_root))

"""
extract_data_helper.save_as_json(extract_data_helper.compute_frequencies(tokenized_text, 2), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/2grams.json')
extract_data_helper.save_as_json(extract_data_helper.compute_frequencies(tokenized_text, 3), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/3grams.json')
extract_data_helper.save_as_json(extract_data_helper.compute_frequencies(tokenized_text, 4), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/4grams.json')
extract_data_helper.save_as_json(extract_data_helper.compute_frequencies(tokenized_text, 5), '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/5grams.json')
"""