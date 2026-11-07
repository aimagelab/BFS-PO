#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path


DEFAULT_CONFIG_NAME = "Python Debugger: Current File"
DEFAULT_DEBUG_HOST = "127.0.0.1"
DEFAULT_DEBUG_PORT = "5678"


def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    escape = False

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        result.append(char)
        i += 1

    json_text = "".join(result)
    return re.sub(r",\s*([}\]])", r"\1", json_text)


def load_launch_config(path, name):
    with path.open() as f:
        data = json.loads(strip_jsonc(f.read()))

    for config in data.get("configurations", []):
        if config.get("name") == name:
            return config

    names = [config.get("name", "<unnamed>") for config in data.get("configurations", [])]
    raise SystemExit(f"Launch configuration '{name}' not found. Available: {', '.join(names)}")


def expand_vscode_vars(value, workspace_folder, active_file):
    if not isinstance(value, str):
        return value

    replacements = {
        "${workspaceFolder}": str(workspace_folder),
        "${file}": str(active_file),
        "${fileBasename}": active_file.name,
        "${fileDirname}": str(active_file.parent),
    }
    for key, replacement in replacements.items():
        value = value.replace(key, replacement)

    def replace_env(match):
        return os.environ.get(match.group(1), "")

    return re.sub(r"\$\{env:([^}]+)\}", replace_env, value)


def build_env(config, workspace_folder, active_file, debug_host, debug_port):
    env = os.environ.copy()
    for key, value in config.get("env", {}).items():
        env[key] = expand_vscode_vars(value, workspace_folder, active_file)

    env["ENABLE_DEBUGPY"] = "1"
    env["DEBUGPY_HOST"] = debug_host
    env["DEBUGPY_PORT"] = debug_port
    env.setdefault("DEBUGPY_WAIT_FOR_CLIENT", "1")
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Run a VS Code Python launch configuration under debugpy for later attach."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="Name of the launch.json configuration to run.")
    parser.add_argument("--launch-json", default=".vscode/launch.json", help="Path to launch.json.")
    parser.add_argument("--file", default="main_grpo.py", help="Value to use for VS Code's ${file} variable.")
    parser.add_argument("--debug-host", default=DEFAULT_DEBUG_HOST, help="Host passed to DEBUGPY_HOST.")
    parser.add_argument("--debug-port", default=DEFAULT_DEBUG_PORT, help="Port passed to DEBUGPY_PORT.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    args = parser.parse_args()

    workspace_folder = Path.cwd().resolve()
    launch_json = Path(args.launch_json)
    if not launch_json.is_absolute():
        launch_json = workspace_folder / launch_json

    active_file = Path(args.file)
    if not active_file.is_absolute():
        active_file = workspace_folder / active_file
    active_file = active_file.resolve()

    config = load_launch_config(launch_json, args.config)
    program = expand_vscode_vars(config.get("program", str(active_file)), workspace_folder, active_file)
    python = expand_vscode_vars(config.get("python", sys.executable), workspace_folder, active_file)
    config_args = [
        expand_vscode_vars(value, workspace_folder, active_file)
        for value in config.get("args", [])
    ]

    command = [python, program, *config_args]
    env = build_env(config, workspace_folder, active_file, args.debug_host, args.debug_port)

    print("Running launch configuration:", args.config, flush=True)
    print("Command:", shlex.join(command), flush=True)
    print(f"Debug attach target: {args.debug_host}:{args.debug_port}", flush=True)

    if args.dry_run:
        return

    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
