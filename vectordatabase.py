import chromadb

class VectorDatabase:
    """Only responsible for DB operations — insert and query"""

    def __init__(self, embedding_function):
        self.client             = chromadb.Client()
        self.collection         = None
        self.embedding_function = embedding_function   # injected, easy to swap

    def create_collection(self, collection_name):
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def insert(self, chunks):
        documents = []
        metadatas = []
        ids       = []

        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk["text"])
            metadatas.append({
                "page":     chunk["page"],
                "start_id": chunk["start_id"],
                "end_id":   chunk["end_id"]
            })

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Inserted {len(chunks)} chunks")

    def query(self, query_text, n_results=3):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )