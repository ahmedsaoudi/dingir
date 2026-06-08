import os
import shutil
import subprocess


def read_file(filepath: str) -> str:
    """Reads and returns the full text contents of a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    """Writes text content to a file, creating parent directories if needed. Overwrites any existing content."""
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
    """Replaces a contiguous range of lines in a file with new content. Lines are 1-indexed and the range is inclusive (both start_line and end_line are replaced). To insert lines without removing any, set start_line and end_line to the same value and include the original line along with the new lines in new_content."""
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
    """Lists all files and subdirectories inside the given directory path."""
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


def get_cwd() -> str:
    """Returns the absolute path of the current working directory where the agent operates."""
    try:
        return os.getcwd()
    except Exception as e:
        return f"Error retrieving current working directory: {str(e)}"


def edit_file(filepath: str, target_content: str, replacement_content: str) -> str:
    """Edits a file by replacing an exact string of text (target_content) with a new string (replacement_content). The target_content must match the existing file contents exactly."""
    try:
        abs_path = os.path.abspath(filepath)
        
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if target_content not in content:
            return (
                f"Error: The provided `target_content` was not found in {filepath}. "
                "Ensure that indentation and whitespace match exactly."
            )
            
        occurrences = content.count(target_content)
        if occurrences > 1:
            return (
                f"Error: The `target_content` was found {occurrences} times in {filepath}. "
                "Please provide a larger chunk of code to ensure a unique match."
            )
            
        new_content = content.replace(target_content, replacement_content)
        
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Successfully edited {filepath}."
        
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


def delete_path(filepath: str) -> str:
    """Deletes a file or directory at the given path."""
    try:
        abs_path = os.path.abspath(filepath)
        if not os.path.exists(abs_path):
            return f"Error: Path not found: {filepath}"
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
            return f"Successfully deleted directory {filepath}"
        else:
            os.remove(abs_path)
            return f"Successfully deleted file {filepath}"
    except Exception as e:
        return f"Error deleting {filepath}: {str(e)}"


def move_path(source: str, destination: str) -> str:
    """Moves or renames a file or directory from source to destination."""
    try:
        abs_src = os.path.abspath(source)
        abs_dst = os.path.abspath(destination)
        if not os.path.exists(abs_src):
            return f"Error: Source not found: {source}"
        shutil.move(abs_src, abs_dst)
        return f"Successfully moved {source} to {destination}"
    except Exception as e:
        return f"Error moving {source}: {str(e)}"


def copy_path(source: str, destination: str) -> str:
    """Copies a file or directory from source to destination."""
    try:
        abs_src = os.path.abspath(source)
        abs_dst = os.path.abspath(destination)
        if not os.path.exists(abs_src):
            return f"Error: Source not found: {source}"
        if os.path.isdir(abs_src):
            shutil.copytree(abs_src, abs_dst)
            return f"Successfully copied directory {source} to {destination}"
        else:
            shutil.copy2(abs_src, abs_dst)
            return f"Successfully copied file {source} to {destination}"
    except Exception as e:
        return f"Error copying {source}: {str(e)}"


def search_files(dirpath: str, query: str) -> str:
    """Searches INSIDE the text content of files for a regex pattern (like 'grep'). Does NOT search file names."""
    try:
        abs_dir = os.path.abspath(dirpath)
        if not os.path.isdir(abs_dir):
            return f"Error: Directory not found: {dirpath}"
        result = subprocess.run(
            ["grep", "-rnI", query, abs_dir],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout if result.stdout else "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Error running grep: {result.stderr}"
    except Exception as e:
        return f"Error searching files: {str(e)}"


def find_paths(dirpath: str, name_pattern: str) -> str:
    """Searches for file or directory NAMES matching a pattern (like '*.py'). Does NOT search inside the file contents."""
    try:
        import fnmatch
        abs_dir = os.path.abspath(dirpath)
        if not os.path.isdir(abs_dir):
            return f"Error: Directory not found: {dirpath}"
            
        matches = []
        for root, dirs, files in os.walk(abs_dir):
            for name in files + dirs:
                if fnmatch.fnmatch(name, name_pattern):
                    full_path = os.path.join(root, name)
                    matches.append(full_path)
                    
        if not matches:
            return f"No paths found matching '{name_pattern}' in {dirpath}."
            
        return "\n".join(sorted(matches))
    except Exception as e:
        return f"Error finding paths: {str(e)}"

