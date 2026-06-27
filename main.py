import glob
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv


def run_bash_command(command: str) -> tuple[str, bool]:
    dangerous_commands = ["su", "sudo", "reboot", "shutdown"]

    if any(dangerous_command in command for dangerous_command in dangerous_commands):
        return "Error: Dangerous command blocked", False

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=20
        )
        output = f"Exit code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"  # noqa: E501
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "Error: Timeout (20s)", False
    except Exception as e:
        return f"Error: {e}", False


def ensure_workdir_path(relative_path: str) -> Path:
    resolved_path = (WORKDIR / relative_path).resolve()

    if not resolved_path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workdir: {relative_path}")

    return resolved_path


def glob_files(pattern: str) -> tuple[str, bool]:
    try:
        matched_files = []

        for matched_path in glob.glob(pattern, root_dir=WORKDIR):
            matched_files.append(matched_path)

        return "\n".join(matched_files) if matched_files else "(no matches)", True
    except Exception as e:
        return f"Error: {e}", False


def read_text_file(file_path: str, max_lines: int | None = None) -> tuple[str, bool]:
    try:
        lines = ensure_workdir_path(file_path).read_text().splitlines()

        if max_lines is not None and 0 < max_lines < len(lines):
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]

        return "\n".join(lines), True
    except Exception as e:
        return f"Error: {e}", False


def write_text_file(file_path: str, text: str) -> tuple[str, bool]:
    try:
        resolved_path = ensure_workdir_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_path.write_text(text)
        return f"Wrote to {file_path}", True
    except Exception as e:
        return f"Error: {e}", False


def replace_first_in_file(
    file_path: str, old_text: str, new_text: str
) -> tuple[str, bool]:
    try:
        resolved_path = ensure_workdir_path(file_path)
        text = resolved_path.read_text()

        if old_text not in text:
            return f"Error: text not found in {file_path}", False

        resolved_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {file_path}", True
    except Exception as e:
        return f"Error: {e}", False


WORKDIR = Path.cwd()

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
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_text_file",
        "description": "Read text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "max_lines": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_text_file",
        "description": "Write text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["file_path", "text"],
        },
    },
    {
        "name": "edit_text_file",
        "description": "Replace the first occurrence of old_text with new_text in the file.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., tuple[str, bool]]] = {
    "bash": run_bash_command,
    "glob": glob_files,
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "edit_text_file": replace_first_in_file,
}


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
                system=f"You are a personal ai assistant at {WORKDIR}. Use tools to solve tasks. Act, don't explain.",  # noqa: E501
                tools=TOOLS,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results: list[ToolResultBlockParam] = []

            for block in response.content:
                if block.type == "tool_use":
                    handler = TOOL_HANDLERS[block.name]
                    value, ok = handler(**block.input)

                    if block.name == "bash":
                        prompt = f"$ {block.input['command']}"
                    else:
                        prompt = f"[{block.name}] {block.input}"

                    prompt_color = "33" if ok else "31"  # Yellow, Red
                    print(f"\033[{prompt_color}m{prompt}\033[0m")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": value,
                        }
                    )

                    for line in value.splitlines():
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
