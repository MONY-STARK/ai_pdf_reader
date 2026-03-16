import fitz

class Highlighter:
    """Only responsible for highlighting PDF"""

    def __init__(self, pages_data):
        self.pages_data = pages_data

    def highlight(self, pdf_path, query_results, output_path="highlighted.pdf"):
        doc  = fitz.open(pdf_path)
        gold = (1, 1, 0)

        for meta in query_results["metadatas"][0]:
            start_id = meta["start_id"]
            end_id   = meta["end_id"]
            page_num = meta["page"]

            matched_words = [
                w for w in self.pages_data[page_num]["words"]
                if start_id <= w["word_id"] <= end_id
            ]

            page = doc[page_num]
            for word in matched_words:
                x0, y0, x1, y1 = word["coords"]
                rect            = fitz.Rect(x0, y0, x1, y1)
                highlight       = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=gold)
                highlight.update()

        doc.save(output_path)
        print(f"Saved: {output_path}")