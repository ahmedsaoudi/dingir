import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from dingir.agents.stdtools.system import calculator, current_datetime
from dingir.agents.stdtools.fs import read_file, write_file, list_directory
from dingir.agents.stdtools.web import web_search, fetch_webpage


# --- Calculator tests ---


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator("2+3") == "5"
        assert calculator("10-4") == "6"
        assert calculator("3*7") == "21"
        assert calculator("15/4") == "3.75"

    def test_operator_precedence(self):
        assert calculator("2+3*4") == "14"
        assert calculator("(2+3)*4") == "20"

    def test_exponentiation_and_modulo(self):
        assert calculator("2**10") == "1024"
        assert calculator("17%5") == "2"

    def test_floor_division(self):
        assert calculator("7//2") == "3"

    def test_unary_operators(self):
        assert calculator("-5") == "-5"
        assert calculator("-5+10") == "5"

    def test_float_expressions(self):
        assert calculator("3.14*2") == "6.28"

    def test_rejects_function_calls(self):
        result = calculator("__import__('os').system('whoami')")
        assert "error" in result.lower()

    def test_rejects_variables(self):
        result = calculator("x + 1")
        assert "error" in result.lower()

    def test_rejects_strings(self):
        result = calculator("'hello' + 'world'")
        assert "error" in result.lower()

    def test_division_by_zero(self):
        result = calculator("1/0")
        assert "error" in result.lower()


# --- current_datetime tests ---


class TestCurrentDatetime:
    def test_returns_string(self):
        result = current_datetime()
        assert isinstance(result, str)

    def test_contains_date_components(self):
        result = current_datetime()
        # Should have YYYY-MM-DD format
        parts = result.split(" ")
        assert len(parts) >= 2
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3


# --- File system tests ---


class TestFileSystem:
    def test_read_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = read_file(str(test_file))
        assert result == "hello world"

    def test_write_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = str(tmp_path / "output.txt")
        result = write_file(target, "some content")
        assert "Successfully" in result
        assert open(target).read() == "some content"

    def test_write_file_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = str(tmp_path / "sub" / "dir" / "file.txt")
        result = write_file(target, "nested")
        assert "Successfully" in result
        assert open(target).read() == "nested"

    def test_list_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = list_directory(str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir" in result
        assert "[DIR]" in result  # subdir should be marked

    def test_list_empty_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        result = list_directory(str(empty))
        assert "Empty" in result

    def test_sandbox_blocks_read_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read_file("/etc/passwd")
        assert "Access Denied" in result

    def test_sandbox_blocks_write_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("/tmp/evil.txt", "pwned")
        assert "Access Denied" in result

    def test_sandbox_blocks_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read_file("../../etc/passwd")
        assert "Access Denied" in result

    def test_sandbox_blocks_list_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = list_directory("/etc")
        assert "Access Denied" in result


# --- Web tools tests (mocked) ---


class TestWebSearch:
    @patch("dingir.agents.stdtools.web.requests.get")
    def test_returns_parsed_snippets(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <a class="result__snippet" href="#">First result snippet</a>
        <a class="result__snippet" href="#">Second result snippet</a>
        """
        mock_get.return_value = mock_response
        result = web_search("test query")
        assert "1." in result
        assert "First result snippet" in result

    @patch("dingir.agents.stdtools.web.requests.get")
    def test_handles_no_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>No matching results</body></html>"
        mock_get.return_value = mock_response
        result = web_search("obscure query")
        assert "No results found" in result

    @patch("dingir.agents.stdtools.web.requests.get")
    def test_handles_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        result = web_search("test")
        assert "503" in result

    @patch("dingir.agents.stdtools.web.requests.get")
    def test_handles_network_exception(self, mock_get):
        mock_get.side_effect = ConnectionError("Network unreachable")
        result = web_search("test")
        assert "error" in result.lower()


class TestFetchWebpage:
    @patch("dingir.agents.stdtools.web.requests.get")
    def test_fetch_webpage_pipeline(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><head><style>body {color: red;}</style></head>"
            "<body>"
            "<header>Header Content</header>"
            "<nav>Navigation</nav>"
            "<p>Hello <b>world</b></p>"
            '<a href="/about">About Us</a>'
            '<img src="tree.jpg" alt="A nice tree">'
            '<img src="empty.jpg">'
            '<script>alert("xss")</script>'
            "</body></html>"
        )
        mock_get.return_value = mock_response
        result = fetch_webpage("http://example.com/home")
        
        # Style and script elements are removed
        assert "color: red;" not in result
        assert "alert" not in result
        
        # Header and nav are removed
        assert "Header Content" not in result
        assert "Navigation" not in result
        
        # Paragraph text is present
        assert "Hello" in result
        assert "world" in result
        
        # Link URL is preserved next to link text with resolved absolute path
        assert "About Us (http://example.com/about)" in result
        
        # Image with alt is replaced, image without alt is decomposed
        assert "[Image: A nice tree]" in result
        
        # Collapsed consecutive empty lines/spaces
        assert "  " not in result

    @patch("dingir.agents.stdtools.web.requests.get")
    def test_handles_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        result = fetch_webpage("http://example.com/missing")
        assert "404" in result



