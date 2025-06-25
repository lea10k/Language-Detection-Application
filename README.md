# Language Detection Application

A web-based language detection tool built in Python, with a user-friendly interface (HTML/CSS) and backend logic for detecting the language of input text. The application uses word-based analysis for (currently) English, Italian and German language.

## How does the website work?

- Enter text in English, Italian, or German into the input field on the left side of the page.
- Click the Submit button to process your input.
- The application will automatically detect the language(s) present in your text.
- Detected languages and results will be displayed below the input field.

## Features

- Detects the language of each word in the input
- Uses n-gram and mean distance algorithms
- Web interface for user interaction

## Constraints & Limitations

- Only supports languages included in the provided language profiles (`json_data/`)
- Detection is optimized for word-level input; sentence-level or longer text will not improve accuracy
- Shorter texts (single ambiguous words) may reduce accuracy, as the model only uses a context window for disambiguation when needed
- Relies on NLTK and NumPy; ensure compatible versions are installed
- Requires Python 3.8+; not tested on other versions
- Application must be started from the project root directory
- The web server runs on localhost:8000 by default (configurable in `app.py`)
- For word-level detection, only Unicode word tokens are supported (special symbols/punctuation may not be detected)
- NLTK may require you to download additional datasets on first run

## Key Project Files and Structure

- **app.py**: The main entry point for the web application (Flask-based). It manages routing and integrates the language detection logic with the web interface.

- **scripts/:**
  This directory contains helper and processing scripts, including:
   - detection_helper.py: Core functions for language detection logic.
   - n_gram.py: Implements n-gram extraction for language modeling.
   - preprocessing.py: Cleans and prepares input text.
   - word_level_detector.py: Performs word-level language detection.
   - Initialize_json_data.py, n_gram_frequencies.py, perplexity.py, etc.

- **data/** and **json_data/:** Directories for data resources and preprocessed language profiles/models.

- **static/:** Contains static web files like CSS stylesheets and images.

- **templates/:**
  Stores HTML template files used by Flask to generate the web interface.

For a detailed list and descriptions of all scripts, see the scripts directory.

## Installation

### Prerequisites

- Python 3.8+ (recommended)
- pip (Python package manager)
- Linux/Debian terminal, WSL, or similar (Windows users: use WSL or compatible terminal)
- (Optional) Visual Studio Code (VS Code)

### Install Required Packages

```bash
pip install flask numpy nltk
```

### (Optional) Use a Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

## Launch & Usage

### Start the Application

From your project directory:

```bash
python app.py
```

**Note:** Do not close this terminal window while the server is running.

### Test the Server

- Open a new terminal in your project directory.
- (Optional) Activate your virtual environment again, if used.
- Test server response:
  ```bash
  curl localhost:8000
  ```
- Or open your browser to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### (Optional) Open with VS Code

If using a virtual environment:
```bash
code .
```

VS Code should detect your virtual environment automatically.

**Note:** On first use, NLTK may prompt for additional dataset downloads. Follow on-screen instructions or consult the NLTK documentation.

## Project Structure

- `app.py`: Main entry point for the web application
- `scripts/`: Helper and processing scripts
- `data/`, `json_data/`: Language resources and models
- `static/`, `templates/`: Front-end files (HTML, CSS, etc.)

## License

This project is released under the MIT License.
