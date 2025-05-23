from flask import Flask, request, render_template  # Import Flask framework and required modules

app = Flask(__name__)  # Create a new Flask web application instance

@app.route("/", methods=["GET"])  # Define route for the homepage, accepts GET requests
def index():
    return render_template("index.html")  # Render and return the 'index.html' template

@app.route("/detect", methods=["POST"])  # Define route for '/detect', accepts POST requests
def detect():
    text = request.form["submission"]  # Get the value of the 'submission' field from the form data
    return f"You sent: {text}"  # Return a response displaying the submitted text

if __name__ == "__main__":  # Check if this script is being run directly
    app.run(debug=True, port=8000)  # Start the Flask development server with debug mode on port 8000