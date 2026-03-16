import fitz

class DataLoader:
    """Only responsible for loading PDF and extracting words"""

    def __init__(self):
        self.docs = None

    def load_data(self, file_path):
        self.docs = fitz.open(file_path)
        return self._get_words()

    def _get_words(self):
        word_id    = 0
        pages_data = []

        for page_num, page in enumerate(self.docs):
            words      = page.get_text("words")
            page_words = []

            for word in words:
                page_words.append({
                    "word_id": word_id,
                    "text":    word[4],
                    "coords":  (word[0], word[1], word[2], word[3]),
                    "page":    page_num
                })
                word_id += 1

            pages_data.append({
                "page_num":  page_num,
                "words":     page_words,
                "full_text": " ".join(w["text"] for w in page_words)
            })

        return pages_data
