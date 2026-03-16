from preprocesser import TextPreprocessor


class Chunker:
    """Only responsible for chunking pages into segments"""

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor   # dependency injected

    def chunk(self, pages_data, chunk_len=150):
        all_chunks = []

        for page in pages_data:
            current_chunk_words = []

            for word in page["words"]:
                current_chunk_words.append(word)

                if word["text"].endswith((".", "?", "!")) and len(current_chunk_words) >= chunk_len:
                    all_chunks.append(self._build_chunk(current_chunk_words, page["page_num"]))
                    current_chunk_words = []

            if current_chunk_words:
                all_chunks.append(self._build_chunk(current_chunk_words, page["page_num"]))

        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks

    def _build_chunk(self, chunk_words, page_num):
        return {
            "text":     self.preprocessor.clean_text(" ".join(w["text"] for w in chunk_words)),
            "words":    chunk_words,
            "start_id": chunk_words[0]["word_id"],
            "end_id":   chunk_words[-1]["word_id"],
            "page":     page_num,
            "coords":   [w["coords"] for w in chunk_words]
        }