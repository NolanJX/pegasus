import glob
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain.tools import tool


@tool("bash", response_format="content_and_artifact")
def run_bash_command(command: str) -> tuple[str, bool]:
    """Run a bash command."""
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


@tool("glob", response_format="content_and_artifact")
def glob_files(pattern: str) -> tuple[str, bool]:
    """Find files matching a glob pattern."""
    try:
        matched_files = []

        for matched_path in glob.glob(pattern, root_dir=WORKDIR):
            matched_files.append(matched_path)

        return "\n".join(matched_files) if matched_files else "(no matches)", True
    except Exception as e:
        return f"Error: {e}", False


@tool(response_format="content_and_artifact")
def read_text_file(file_path: str, max_lines: int | None = None) -> tuple[str, bool]:
    """Read text file."""
    try:
        lines = ensure_workdir_path(file_path).read_text().splitlines()

        if max_lines is not None and 0 < max_lines < len(lines):
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]

        return "\n".join(lines), True
    except Exception as e:
        return f"Error: {e}", False


@tool(response_format="content_and_artifact")
def write_text_file(file_path: str, text: str) -> tuple[str, bool]:
    """Write text file."""
    try:
        resolved_path = ensure_workdir_path(file_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_path.write_text(text)
        return f"Wrote to {file_path}", True
    except Exception as e:
        return f"Error: {e}", False


@tool("edit_text_file", response_format="content_and_artifact")
def replace_first_in_file(
    file_path: str, old_text: str, new_text: str
) -> tuple[str, bool]:
    """Replace the first occurrence of old_text with new_text in the file."""
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

TOOLS = [
    run_bash_command,
    glob_files,
    read_text_file,
    write_text_file,
    replace_first_in_file,
]


def main():
    load_dotenv(override=True)

    model_id = os.getenv("MODEL_ID")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if model_id is None or base_url is None or api_key is None:
        sys.exit(os.EX_CONFIG)

    agent = create_agent(
        model=init_chat_model(model_id, max_tokens=1000),
        tools=TOOLS,
        system_prompt=f"You are a personal ai assistant at {WORKDIR}. Use tools to solve tasks. Act, don't explain.",  # noqa: E501
    )
    messages: list[AnyMessage] = []

    while True:
        message = input("\033[36m>> \033[0m")  # Cyan

        if message.strip().lower() == "q":
            break

        messages.append(HumanMessage(message))

        pending_tool_prompts = {}
        stream = agent.stream_events({"messages": messages}, version="v3")

        for snapshot in stream.values:
            messages = snapshot["messages"]
            latest_message = messages[-1]

            if isinstance(latest_message, HumanMessage):
                continue

            if isinstance(latest_message, AIMessage):
                if latest_message.tool_calls:
                    for tool_call in latest_message.tool_calls:
                        if tool_call["name"] == "bash":
                            prompt = f"$ {tool_call['args']['command']}"
                        else:
                            prompt = f"[{tool_call['name']}] {tool_call['args']}"

                        pending_tool_prompts[tool_call["id"]] = prompt
                else:
                    for content_block in latest_message.content_blocks:
                        if content_block["type"] == "reasoning":
                            print(f"\033[90;3m{content_block['reasoning']}\033[0m")
                        elif content_block["type"] == "text":
                            print(content_block["text"])
            elif isinstance(latest_message, ToolMessage):
                value = latest_message.content
                ok = latest_message.artifact

                prompt = pending_tool_prompts.pop(latest_message.tool_call_id)
                prompt_color = "33" if ok else "31"  # Yellow, Red
                print(f"\033[{prompt_color}m{prompt}\033[0m")

                for line in value.splitlines():
                    print(f"\033[90m| {line}\033[0m")  # Gray


if __name__ == "__main__":
    main()
