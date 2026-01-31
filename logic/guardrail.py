from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Guardrail:
    def __init__(
        self, client: OpenAI, embedding_model: str, reference_topics: list, max_buffer_size: int = 10000
    ):
        # Setting a maxlen prevents the buffer from consuming infinite memory
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.client: OpenAI = client
        self.embedding_model: str = embedding_model
        self.reference_topics: list = reference_topics

    def _upload_buffer(self, item):
        self.buffer.append(item)

    def __get_embedding(self, text):
        response = self.client.embeddings.create(
            model=self.embedding_model, input=text, encoding_format="float"
        ).to_dict()

        return {
            "text": text,
            "embedding": response["data"][0]["embedding"][:5],
            "total_tokens": response["usage"]["total_tokens"],
        }

    def __clear_buffer(self):
        self.buffer.clear()

    def __accumulator(self, token, strategy: str = "full_stop"):
        self.buffer.append(token)

        if strategy == "full_stop":
            if "." in token:
                buffer_list = list(self.buffer)
                self.__clear_buffer()
                return " ".join(buffer_list)
            

    def generate_topic_embeddings(self):
        topic_emb = [self.__get_embedding(topic) for topic in self.reference_topics]
        return topic_emb
    
    
    def process(self, token, ):
        chunk_of_tokens = self.__accumulator(token)

        if chunk_of_tokens:
            response = self.__get_embedding(chunk_of_tokens)
            embedding = response["embedding"]
            total_tokens = response["total_tokens"]

            res_dict = {
                "chunk": chunk_of_tokens,
                "embedding": embedding,
                "total_tokens": total_tokens,
            }

            print(res_dict)

        


# --- Execution ---
load_dotenv()

texts = ["Io ", "vivo ", "a ", "New ", "York.", " Mi ", "piace ", "il ", "calcio."]

gg = Guardrail(embedding_model="text-embedding-3-small", client=OpenAI(), reference_topics=["New York", "Vivere a New York"])
top_ref = gg.generate_topic_embeddings()
print(top_ref)

print("--- Processing Results ---")
for token in texts:
    result = gg.process(token=token)
    if result:
        print(" - ", result)
