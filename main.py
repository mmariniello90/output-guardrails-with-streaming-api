from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.style import Style
from rich.live import Live
from rich.text import Text
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_text(text_path: str) -> str:
    with open(text_path, "r") as file:
        content = file.read()
    return content


def call_model(client: OpenAI, text: str, model: str) -> dict:
    response = client.embeddings.create(
        model=model, input=text, encoding_format="float"
    ).to_dict()

    return {
        "embedding": response["data"][0]["embedding"],
        "total_tokens": response["usage"]["total_tokens"],
    }


def split_text(text, limit, separator=None):
    if separator:
        return text.split(separator)
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def main():
    # Initialize variables and environment
    load_dotenv()
    client = OpenAI()

    EMB_MODEL = "text-embedding-3-small"

    console = Console()
    danger_style = Style(color="#e83d73", blink=True, bold=True)
    safe_style = Style(color="#2a7f62", blink=True, bold=True)
    neutral_style = Style(color="white")  # colore durante lo streaming parola per parola

    # Reference topics
    reference_text = ["Description of New York City"]

    ref_emb = [
        call_model(client=client, text=text, model=EMB_MODEL)["embedding"]
        for text in reference_text
    ]
    reference_embedding = {"text": reference_text, "embedding": ref_emb}

    content = load_text("data/SCALA_CODING.txt")
    split_text_list = split_text(content, 0, separator=".")

    previous_chunks = []
    total_tokens_count = []

    with Live("", console=console, refresh_per_second=20) as live:
        for chunk in split_text_list:
            # 1. Get the embedding and distance first
            response = call_model(client=client, text=chunk, model=EMB_MODEL)
            chunk_embedding = response["embedding"]
            target = np.array(chunk_embedding).reshape(1, -1)
            reference = np.array(reference_embedding["embedding"])
            distance = cosine_similarity(target, reference)[0][0]

            words = chunk.split()
            streamed_chunk = Text()

            # 2. Typing Loop: Always uses neutral_style
            for word in words:
                streamed_chunk.append(word + " ", style=neutral_style)

                display_text = Text()
                for prev_text, prev_dist in previous_chunks:
                    style = danger_style if prev_dist > 0.3 else safe_style
                    display_text.append(prev_text, style=style)

                display_text.append(streamed_chunk)
                live.update(display_text)
                time.sleep(0.05)

            # 3. SNAP: Now that the loop is over, add to history
            previous_chunks.append((" ".join(words) + ". ", distance))
            
            # 4. REFRESH: Force the UI to re-render the history including this new colored chunk
            final_display = Text()
            for prev_text, prev_dist in previous_chunks:
                style = danger_style if prev_dist > 0.3 else safe_style
                final_display.append(prev_text, style=style)
            
            live.update(final_display)

            total_tokens_count.append(response["total_tokens"])

    print(f"Total tokens: {sum(total_tokens_count)}")
    print(f"Total cost: {(0.02/1_000_000) * sum(total_tokens_count)}")


if __name__ == "__main__":
    main()
