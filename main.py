import os
import subprocess
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

TOOLS = [
    {
        "name": "bash",
        "description": "Run a Bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        }
    }
]


def run_bash_command(command: str) -> str:
    dangerous_commands = ["su", "sudo", "reboot", "shutdown"]

    if any(dangerous_command in command for dangerous_command in dangerous_commands):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
        return f"Exit code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (20s)"
    except Exception as e:
        return f"Error: {e}"


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
                system="You are a personal ai assistant. Use Bash to solve tasks. Act, don't explain.",
                tools=TOOLS,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"\033[34m$ {block.input['command']}\033[0m")

                    command_output = run_bash_command(block.input["command"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": command_output,
                    })

                    print(command_output)

            messages.append({"role": "user", "content": tool_results})

        response_content = messages[-1]["content"]

        for block in response_content:
            if block.type == "thinking":
                print(f"\033[90;3m{block.thinking}\033[0m")  # Gray
            elif block.type == "text":
                print(block.text)


if __name__ == "__main__":
    main()
