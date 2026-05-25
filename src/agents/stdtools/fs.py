import os


def _is_safe_path(path: str) -> bool:
    """Checks whether a given path resolves inside the current working directory."""
    abs_path = os.path.abspath(path)
    cwd = os.getcwd()
    return abs_path.startswith(cwd)


def read_file(filepath: str) -> str:
    """Reads and returns the full text contents of a file. The file must be inside the current working directory."""
    if not _is_safe_path(filepath):
        return "Access Denied: Path is outside the permitted workspace directory."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    """Writes text content to a file, creating parent directories if needed. The file must be inside the current working directory. Overwrites any existing content."""
    if not _is_safe_path(filepath):
        return "Access Denied: Path is outside the permitted workspace directory."
    try:
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def list_directory(dirpath: str = ".") -> str:
    """Lists all files and subdirectories inside the given directory path. The directory must be inside the current working directory."""
    if not _is_safe_path(dirpath):
        return "Access Denied: Path is outside the permitted workspace directory."
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
