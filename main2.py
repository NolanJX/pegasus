import glob
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command


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


@wrap_tool_call
def print_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_call = request.tool_call

    if tool_call["name"] == "bash":
        prompt = f"$ {tool_call['args']['command']}"
    else:
        prompt = f"[{tool_call['name']}] {tool_call['args']}"
    print(f"\033[33m{prompt}\033[0m")  # Yellow

    result = handler(request)

    if isinstance(result, ToolMessage):
        value, ok = result.content, result.artifact
        value_color = "90" if ok else "31"  # Gray, Red

        for line in str(value).splitlines():
            print(f"\033[{value_color}m| {line}\033[0m")

    return result


class ApprovalRule(TypedDict):
    tools: list[str]
    match: Callable[[dict[str, Any]], bool]
    reason: str


APPROVAL_RULES: list[ApprovalRule] = [
    {
        "tools": ["bash"],
        "match": lambda tool_input: any(
            restricted_command in tool_input["command"] for restricted_command in ["rm"]
        ),
        "reason": "Restricted command",
    },
    {
        "tools": ["write_text_file", "edit_text_file"],
        "match": lambda _tool_input: True,
        "reason": "File write operation",
    },
]


def approval_reason(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    for approval_rule in APPROVAL_RULES:
        if tool_name in approval_rule["tools"] and approval_rule["match"](tool_input):
            return approval_rule["reason"]
    return None


@wrap_tool_call
def approve_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_call = request.tool_call
    reason = approval_reason(tool_call["name"], tool_call["args"])
    if reason is not None:
        print(f"\033[35m{reason}\033[0m")  # Magenta

        if input("\033[36mAllow? [y/N] \033[0m").strip().lower() != "y":  # Cyan
            return ToolMessage(
                "Error: Permission denied", tool_call_id=tool_call["id"], artifact=False
            )

    return handler(request)


def agent_loop(
    agent, messages: list[AnyMessage]
) -> tuple[list[AnyMessage], AIMessage | None]:
    final_message: AIMessage | None = None

    seen_message_count = len(messages)

    stream = agent.stream_events({"messages": messages}, version="v3")

    for snapshot in stream.values:
        messages = snapshot["messages"]
        unseen_messages = messages[seen_message_count:]

        for message in unseen_messages:
            if isinstance(message, AIMessage) and not message.tool_calls:
                final_message = message

        seen_message_count = len(messages)

    return messages, final_message


def chat_loop(agent, messages: list[AnyMessage]):
    while True:
        message = input("\033[36m>> \033[0m")  # Cyan

        if message.strip().lower() == "q":
            break

        messages.append(HumanMessage(message))

        messages, final_message = agent_loop(agent, messages)

        if final_message is None:
            continue

        for content_block in final_message.content_blocks:
            if content_block["type"] == "reasoning":
                print(f"\033[90;3m{content_block['reasoning']}\033[0m")
            elif content_block["type"] == "text":
                print(content_block["text"])


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
        middleware=[print_tool_call, approve_tool_call],
    )
    messages: list[AnyMessage] = []

    chat_loop(agent, messages)


if __name__ == "__main__":
    main()
