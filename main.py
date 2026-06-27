import os
import subprocess
import sys
from typing import cast

from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a Bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    }
]


def run_bash_command(command: str) -> tuple[str, int | None]:
    dangerous_commands = ["su", "sudo", "reboot", "shutdown"]

    if any(dangerous_command in command for dangerous_command in dangerous_commands):
        return "Error: Dangerous command blocked", None

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=20
        )
        output = f"Exit code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"  # noqa: E501
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "Error: Timeout (20s)", None
    except Exception as e:
        return f"Error: {e}", None


def main():
    load_dotenv(override=True)

    model_id = os.getenv("MODEL_ID")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if model_id is None or base_url is None or api_key is None:
        sys.exit(os.EX_CONFIG)

    client = Anthropic(api_key=api_key, base_url=base_url)
    messages: list[MessageParam] = []

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
                system="You are a personal ai assistant. Use Bash to solve tasks. Act, don't explain.",  # noqa: E501
                tools=TOOLS,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results: list[ToolResultBlockParam] = []

            for block in response.content:
                if block.type == "tool_use":
                    command = cast(str, block.input["command"])
                    command_output, exit_code = run_bash_command(command)

                    command_color = (
                        "31" if (exit_code is not None and exit_code != 0) else "33"
                    )  # Red, Yellow
                    print(f"\033[{command_color}m$ {command}\033[0m")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": command_output,
                        }
                    )

                    for line in command_output.splitlines():
                        print(f"\033[90m| {line}\033[0m")  # Gray

            messages.append({"role": "user", "content": tool_results})

        response_content = cast(list[ContentBlock], messages[-1]["content"])

        for block in response_content:
            if block.type == "thinking":
                print(f"\033[90;3m{block.thinking}\033[0m")  # Gray
            elif block.type == "text":
                print(block.text)


if __name__ == "__main__":
    main()
