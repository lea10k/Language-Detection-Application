# File: Tutorial/app.py
from flask import Flask

# create an instance of the flask application
app = Flask(__name__)

# Create a route on your app
@app.route("/", strict_slashes=False, methods=["GET"])
def index():
    return "<h1>This is the Home Page</h1>"

if __name__ == '__main__':
    app.run(debug=True, port=5100)
