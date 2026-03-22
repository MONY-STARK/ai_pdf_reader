import re


class TextPreprocessor:
    def __init__(self):
        self.skip = {".", ",", "!", "?", ";", ":", "-", "--", "•", "*"}
 
    def preprocess(self, pages_data):
        for page in pages_data:
            page["words"] = [
                w for w in page["words"]
                if w["text"].strip() not in self.skip
            ]
            page["full_text"] = " ".join(w["text"] for w in page["words"])
        return pages_data
 
    def clean_text(self, text):
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text