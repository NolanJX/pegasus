import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv


def main():
    load_dotenv(override=True)

    model_id = os.getenv("MODEL_ID")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if model_id is None or base_url is None or api_key is None:
        sys.exit(os.EX_CONFIG)

    client = Anthropic(api_key=api_key, base_url=base_url)
    messages = []

    while True:
        message = input("\033[36m>> \033[0m")  # Cyan

        if message.strip().lower() == "q":
            break

        messages.append({"role": "user", "content": message})

        while True:
            response = client.messages.create(
                model=model_id,
                max_tokens=1000,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

        response_content = messages[-1]["content"]

        for block in response_content:
            if block.type == "thinking":
                print(f"\033[90;3m{block.thinking}\033[0m")  # Gray
            elif block.type == "text":
                print(block.text)


if __name__ == "__main__":
    main()
