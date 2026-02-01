from logic.guardrail import StreamingGuardrail
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from rich.console import Console
import time


def load_text(text_path: str) -> str:
    with open(text_path, "r") as file:
        content = file.read()
    return content


def get_tokens_from_string(encoder, string: str) -> list:
    tokens = encoder.encode(string)
    return tokens


def main():
    # Initialize variables and environment
    load_dotenv()
    client = OpenAI()
    encoder = tiktoken.get_encoding("cl100k_base")
    console = Console()

    stream_gr = StreamingGuardrail(
        embedding_model="text-embedding-3-small",
        client=client,
        reference_topics=["Living in New York"],
        return_embeddings=False,
    )

    topics_reference = stream_gr.generate_topic_embeddings()

    content = load_text("data/SCALA_CODING.txt")

    for token in get_tokens_from_string(encoder=encoder, string=content):
        token = encoder.decode([token])

        print(token, end="", flush=True)

        result = stream_gr.process(token=token, reference=topics_reference)

        if result:
            similarity = round(result["similarity"][0][0], 3)
            similarity_str = f"[Similarity: {similarity}]"
            console.print(
                similarity_str + "[Blocked!]", style="#f00707"
            ) if similarity > 0.5 else console.print(similarity_str, style="#05a540")

        time.sleep(0.05)


if __name__ == "__main__":
    main()
