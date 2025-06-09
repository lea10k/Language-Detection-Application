import sys
from flask import Flask, request, render_template  # Import Flask framework and required modules
sys.path.append('/home/lea_k/language_detection_project/Language-Detection-Application/scripts')  # Add the scripts directory to the system path for module imports
from word_level_detector_copy import WordLevelLanguageDetectorCopy
from postprocessing import ReplaceProcessedText
from postprocessing import colorize_text  

app = Flask(__name__)  # Create a new Flask web application instance

model_paths = {
    'german': {
        '1gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/1grams.json',
        '2gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/2grams.json',
        '3gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/3grams.json',
        '4gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/4grams.json',
        '5gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/5grams.json'
    },
    'english': {
        '1gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/1grams.json',
        '2gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/2grams.json',
        '3gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/3grams.json',
        '4gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/4grams.json',
        '5gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/english/5grams.json'
    },
    'italian': {
        '1gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/1grams.json',
        '2gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/2grams.json',
        '3gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/3grams.json',
        '4gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/4grams.json',
        '5gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/italian/5grams.json'
    }
}

detector = WordLevelLanguageDetectorCopy(model_paths)

@app.route("/", methods=["GET"])  # Define route for the homepage, accepts GET requests
def index():
    return render_template("index.html")  # Render and return the 'index.html' template

@app.route("/detect", methods=["POST"])  # Define route for '/detect', accepts POST requests
def detect():
    text = request.form["submission"]  # Get the value of the 'submission' field from the form data
    results = detector.detect_text_languages(text)
    final = ReplaceProcessedText(results, text)
    colored_text = colorize_text(final)
    return render_template("index.html", text=text, results=final, colored_text=colored_text)

if __name__ == "__main__":  # Check if this script is being run directly
    app.run(debug=True, port=8000)  # Start the Flask development server with debug mode on port 8000