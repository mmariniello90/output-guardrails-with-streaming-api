from logic.guardrail import StreamingGuardrail
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from rich.console import Console
import time
from logic.streaming_detector import PeakDetector

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

    # Initialize PeakDetector
    # Using absolute_threshold=0.5 to match the original run.py blocking logic
    detector = PeakDetector(window_size=30, z_threshold=3.5, absolute_threshold=0.5)

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
            
            # Use PeakDetector to decide if it's a peak/block
            detection_result = detector.add(similarity)
            is_peak = detection_result["is_peak"]

            if is_peak:
                reason = detection_result["reason"]
                extra_info = ""
                if reason == "z_score":
                    z_val = detection_result.get("z_score")
                    extra_info = f" [Z-Score: {z_val:.1f}]"
                elif reason == "absolute_threshold":
                    extra_info = " [Abs Threshold]"
                
                similarity_str = f"[Similarity: {similarity}{extra_info}]"
                console.print(f"\n{similarity_str} [BLOCKED!]", style="bold #f00707")
            else:
                similarity_str = f"[Similarity: {similarity}]"
                console.print(similarity_str, style="#05a540")

        time.sleep(0.05)


if __name__ == "__main__":
    main()
