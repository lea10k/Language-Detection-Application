def ReplaceProcessedText(results, input_text):
    input_text = input_text.split()
    for i, item in enumerate(results):
        item["word"] = input_text[i] if i < len(input_text) else item["word"]
    return results

#print(ReplaceProcessedText([{"word": "_hello_", "language": "english"}, {"word": "_world_", "language": "english"}], "hello world"))  # Example usage

language_colors = {
    "english": "#FF0000",
    "german": "#00FF00",
    "italian": "#0000FF",
    "ambiguous": "#CCCCCC",
    "unknown": "#999999"
}

def colorize_text(results):
    colored = []
    for item in results:
        word = item["word"].replace("_", "")
        color = language_colors.get(item["language"], "#000000")
        # Erzeuge ein span-Element mit Inline-Style
        colored.append(f'<span style="color: {color};">{word}</span>')
    return " ".join(colored)