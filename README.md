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

## Out-of-Place Distance for Word-Level Language Detection

This application uses the Out-of-Place Distance (OoP) method to classify the language of individual words. For each word to be classified, the OoP distance is calculated against the language profiles of all supported languages. The language with the lowest distance is considered the most likely language of the word.

### Implementation Details

1. **N-Gram Profiles:**  
   Each language has a profile consisting of a ranked list of N-grams (by default, N = 2, 3, 4, 5).

2. **OoP Distance Calculation:**  
   The input word is split into N-grams. For each N-gram, its position (rank) in the profile of each language is checked:
   - If the N-gram exists in the profile, its rank is used.
   - If not found, a high penalty rank (typically 1000) is used.

3. **Distance Normalization:**  
   The sum of the ranks for all N-grams is divided by the number of N-grams (normalization).

4. **Multiple N-Gram Types:**  
   OoP distances are calculated for all configured N-gram lengths. The final result is the average of these values.

5. **Language Selection:**  
   The language with the lowest average OoP distance is assigned to the word.

### Example (Simplified):

- Word: “Hallo”
- Languages: English, German, Italian
- The OoP distance is computed for each language.
- The language with the smallest distance is chosen.

### Code Implementation

The logic is implemented in the `_out_of_place_distance(word, lang)` method of the `WordLevelLanguageDetector` class (`scripts/word_level_detector.py`).  
The actual word detection step then calls the OoP distance computation for all available languages and selects the best match (`detect_word`).

