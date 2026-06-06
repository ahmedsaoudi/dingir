import os


def _is_safe_path(path: str) -> bool:
    """Checks whether a given path resolves inside the current working directory."""
    abs_path = os.path.abspath(path)
    cwd = os.getcwd()
    return abs_path.startswith(cwd)


def read_file(filepath: str) -> str:
    """Reads and returns the full text contents of a file. The file must be inside the current working directory."""
    if not _is_safe_path(filepath):
        return (
            "Access Denied: Path is outside the permitted workspace directory."
        )
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    """Writes text content to a file, creating parent directories if needed. The file must be inside the current working directory. Overwrites any existing content."""
    if not _is_safe_path(filepath):
        return (
            "Access Denied: Path is outside the permitted workspace directory."
        )
    try:
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def replace_lines(
    filepath: str, start_line: int, end_line: int, new_content: str
) -> str:
    """Replaces a contiguous range of lines in a file with new content. Lines are 1-indexed and the range is inclusive (both start_line and end_line are replaced). To insert lines without removing any, set start_line and end_line to the same value and include the original line along with the new lines in new_content. The file must be inside the current working directory."""
    if not _is_safe_path(filepath):
        return (
            "Access Denied: Path is outside the permitted workspace directory."
        )
    try:
        abs_path = os.path.abspath(filepath)
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        if start_line < 1 or end_line < start_line or start_line > total:
            return (
                f"Invalid line range: start_line={start_line}, "
                f"end_line={end_line}, file has {total} lines. "
                f"Lines are 1-indexed and the range must satisfy "
                f"1 <= start_line <= end_line."
            )
        # Clamp end_line to the last line so the model can overshoot safely
        end_line = min(end_line, total)

        # Ensure new_content ends with a newline so it splices cleanly
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        replaced_count = end_line - start_line + 1
        lines[start_line - 1 : end_line] = (
            new_content.splitlines(keepends=True) if new_content else []
        )

        with open(abs_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        new_total = len(lines)
        return (
            f"Replaced lines {start_line}-{start_line + replaced_count - 1} "
            f"({replaced_count} lines) in {filepath}. "
            f"File now has {new_total} lines (was {total})."
        )
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error replacing lines: {str(e)}"


def list_directory(dirpath: str = ".") -> str:
    """Lists all files and subdirectories inside the given directory path. The directory must be inside the current working directory."""
    if not _is_safe_path(dirpath):
        return (
            "Access Denied: Path is outside the permitted workspace directory."
        )
    try:
        items = sorted(os.listdir(dirpath))
        if not items:
            return "(Empty directory)"
        lines = []
        for item in items:
            full = os.path.join(dirpath, item)
            prefix = "[DIR] " if os.path.isdir(full) else "      "
            lines.append(f"{prefix}{item}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
