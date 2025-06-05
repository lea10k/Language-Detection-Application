import sys
from flask import Flask, request, render_template  # Import Flask framework and required modules
sys.path.append('/home/lea_k/language_detection_project/Language-Detection-Application/scripts')  # Add the scripts directory to the system path for module imports
from word_level_detector import WordLevelLanguageDetector

app = Flask(__name__)  # Create a new Flask web application instance

model_paths = {
    'german': {
        '3gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/3grams.json',
        '4gram': '/home/lea_k/language_detection_project/Language-Detection-Application/json_data/german/4grams.json'
    },
    
    #HIER MUSS NOCH ITALIANO UND ENGLISCH HINZUGEFÜGT WERDEN
}

detector = WordLevelLanguageDetector(model_paths)

@app.route("/", methods=["GET"])  # Define route for the homepage, accepts GET requests
def index():
    return render_template("index.html")  # Render and return the 'index.html' template

@app.route("/detect", methods=["POST"])  # Define route for '/detect', accepts POST requests
def detect():
    text = request.form["submission"]  # Get the value of the 'submission' field from the form data
    results = detector.detect_text_languages(text)
    return render_template("index.html", text=text, results=results)

if __name__ == "__main__":  # Check if this script is being run directly
    app.run(debug=True, port=8000)  # Start the Flask development server with debug mode on port 8000