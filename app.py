import sys
sys.path.append('./scripts')

import website_functions
from website_functions import replace_processed_text, colorize_text, count_amount_of_words_of_language

from flask import Flask, request, render_template  
from word_level_detector import WordLevelLanguageDetector

app = Flask(__name__)  

model_paths = {
    'German': {
        '2gram': './json_data/german/2grams.json',
        '3gram': './json_data/german/3grams.json',
        '4gram': './json_data/german/4grams.json',
        '5gram': './json_data/german/5grams.json'
    },
    'English': {
        '2gram': './json_data/english/2grams.json',
        '3gram': './json_data/english/3grams.json',
        '4gram': './json_data/english/4grams.json',
        '5gram': './json_data/english/5grams.json'
    },
    'Italian': {
        '2gram': './json_data/italian/2grams.json',
        '3gram': './json_data/italian/3grams.json',
        '4gram': './json_data/italian/4grams.json',
        '5gram': './json_data/italian/5grams.json'
    }
}

detector = WordLevelLanguageDetector(model_paths)

@app.route("/", methods=["GET"])  # Define route for the homepage, accepts GET requests
def index():
    return render_template("index.html")  # Render and return the 'index.html' template

@app.route("/detect", methods=["POST"])  # Define route for '/detect', accepts POST requests
def detect():
    """
    This function retrieves the text from the form, processes it to detect languages,
    and returns the results to be displayed on the webpage.
    It uses the WordLevelLanguageDetector to analyze the text and returns the processed results
    along with the original text and additional information such as colored text and word counts.
    :return: Rendered HTML template with detection results.
    """
    text = request.form["submission"]  # Get the value of the 'submission' field from the form data
    results = detector.detect_text_languages(text)
    final = replace_processed_text(results, text)
    colored_text = colorize_text(final)
    amount_of_words_in_lang = count_amount_of_words_of_language(results)
    percentage_of_language = website_functions.percentage_of_language(amount_of_words_in_lang)
    return render_template("index.html", text=text, results=final, colored_text=colored_text, amount_of_words_in_lang=amount_of_words_in_lang, 
                           percentage_of_language=percentage_of_language)

if __name__ == "__main__": 
    app.run(debug=True, port=8000)  