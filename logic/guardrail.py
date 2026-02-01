from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class StreamingGuardrail:
    def __init__(
        self,
        client: OpenAI,
        embedding_model: str,
        reference_topics: list,
        return_embeddings: bool,
        max_buffer_size: int = 10000,
    ):
        # Setting a maxlen prevents the buffer from consuming infinite memory
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.client: OpenAI = client
        self.embedding_model: str = embedding_model
        self.reference_topics: list = reference_topics
        self.return_embeddings = False

    def _upload_buffer(self, item):
        self.buffer.append(item)

    def __get_embedding(self, text):
        response = self.client.embeddings.create(
            model=self.embedding_model, input=text, encoding_format="float"
        ).to_dict()

        return {
            "text": text,
            "embedding": response["data"][0]["embedding"],
            "total_tokens": response["usage"]["total_tokens"],
        }

    def __clear_buffer(self):
        self.buffer.clear()

    def __accumulator(self, token, strategy: str = "full_stop"):
        self.buffer.append(token)

        if strategy == "full_stop":
            # Better check: does the token end the sentence?
            # Note: This still has the 'remainder' issue if the token 
            # contains text AFTER the period.
            if any(char in token for char in [".", "!", "?"]):
                buffer_list = list(self.buffer)
                self.__clear_buffer()
                return "".join(buffer_list)
        return None

    def generate_topic_embeddings(self):
        return [self.__get_embedding(topic) for topic in self.reference_topics]

    def process(self, token: str, reference: list):
        chunk_of_tokens = self.__accumulator(token)

        if chunk_of_tokens:
            chunk_response = self.__get_embedding(chunk_of_tokens)
            chunk_embedding = chunk_response["embedding"]
            chunk_total_tokens = chunk_response["total_tokens"]

            reference_text = [topic["text"] for topic in reference]
            reference_embeddings = [topic["embedding"] for topic in reference]

            distances = cosine_similarity(
                np.array(chunk_embedding).reshape(1, -1),
                np.array(reference_embeddings)
            )

            return {
                "chunk": {
                    "chunk_text": chunk_of_tokens,
                    "chunk_embedding": chunk_embedding
                    if self.return_embeddings
                    else None,
                    "chunk_total_tokens": chunk_total_tokens,
                },
                "reference": {
                    "reference_text": reference_text,
                    "reference_embeddings": reference_embeddings
                    if self.return_embeddings
                    else None,
                },
                "similarity": distances,
            }
        else:
            return None
