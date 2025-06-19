from text_processing import preprocess_for_website

def replace_processed_text(results: list[dict], input_text: str) -> dict:
    processed_text = preprocess_for_website(input_text).split()
    for i, item in enumerate(results):
        item["word"] = processed_text[i] if i < len(processed_text) else item["word"]
    return results

language_colors = {
    "English": "#FF0000",
    "German": "#2DDF00",
    "Italian": "#0000FF",
    "Ambiguous": "#676767",
}

def colorize_text(results: dict) -> str:
    colored = []
    for item in results:
        word = item["word"]
        color = language_colors.get(item["language"], "#000000")
        colored.append(f'<span style="color: {color};">{word}</span>')
    return " ".join(colored)
