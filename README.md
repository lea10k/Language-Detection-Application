# Language Detection Application

This project is a language detection application built primarily in Python, with supporting HTML and CSS for the user interface. It is designed to automatically detect the language of a given text using statistical and rule-based methods. The application is organized for easy extensibility and modularity, with scripts handling various aspects of language processing and detection.

## Features

- Detects the language of input text
- Uses n-gram and word-level analysis
- Modular codebase for easy updates and improvements
- Simple web interface

## Project Structure

- `app.py`: The main entry point for the web application (likely a Flask app).
- `scripts/`: Contains all helper and processing scripts.
- `data/`, `json_data/`: Data directories for language resources and models.
- `static/`, `templates/`: Front-end resources (HTML, CSS, etc.).

## Description of Scripts

Here is a summary of the main scripts in the `scripts/` directory (only the first 10 are listed here; to see more, visit the [scripts folder](https://github.com/lea10k/Language-Detection-Application/tree/main/scripts)):

- **[Initialize_json_data.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/Initialize_json_data.py)**: Initializes and prepares language data in JSON format for use in detection algorithms.
- **[detection_helper.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/detection_helper.py)**: Contains functions that assist the main detection logic, such as scoring, matching, and aggregating results.
- **[n_gram.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/n_gram.py)**: Implements n-gram extraction and related utilities for language modeling.
- **[n_gram_frequencies.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/n_gram_frequencies.py)**: Calculates and stores frequency statistics for n-grams across different languages.
- **[perplexity.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/perplexity.py)**: Computes perplexity metrics to evaluate how well a language model predicts a sample.
- **[postprocessing.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/postprocessing.py)**: Handles tasks that occur after detection, such as formatting or filtering results.
- **[preprocessing.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/preprocessing.py)**: Cleans and prepares input text for further analysis and detection.
- **[tokenization.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/tokenization.py)**: Splits input text into tokens (words, characters, etc.) required for detection.
- **[word_level_detector.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/word_level_detector.py)**: Performs language detection at the word level, likely using dictionaries or statistical rules.
- **[word_level_detector_copy.py](https://github.com/lea10k/Language-Detection-Application/blob/main/scripts/word_level_detector_copy.py)**: A variant or backup of the main word-level detector script.

*Note: The above list may be incomplete. To see the full set of scripts, visit the [scripts directory](https://github.com/lea10k/Language-Detection-Application/tree/main/scripts) on GitHub.*
