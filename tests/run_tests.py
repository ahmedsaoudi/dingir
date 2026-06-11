import sys
import inspect
from types import ModuleType

# Mock pytest
pytest_mock = ModuleType("pytest")


class RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"{self.expected_exception.__name__} not raised"
            )
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match and self.match not in str(exc_val):
            raise AssertionError(
                f"Exception message '{str(exc_val)}' does not match pattern '{self.match}'"
            )
        return True


pytest_mock.raises = RaisesContext
sys.modules["pytest"] = pytest_mock

# Add src and root to sys.path
sys.path.insert(0, "./src")
sys.path.insert(0, ".")

# Run the test functions from test_config_composition
from tests.test_config_composition import (
    test_explicit_args_tracking,
    test_config_merge_rightmost_precedence,
    test_config_merge_none_ignored,
    test_invalid_config_type_raises_error,
    test_driver_composite_config,
    test_openai_driver_parameter_conversion,
)

print("Running test_explicit_args_tracking...")
test_explicit_args_tracking()

print("Running test_config_merge_rightmost_precedence...")
test_config_merge_rightmost_precedence()

print("Running test_config_merge_none_ignored...")
test_config_merge_none_ignored()

print("Running test_invalid_config_type_raises_error...")
test_invalid_config_type_raises_error()

print("Running test_driver_composite_config...")
test_driver_composite_config()

print("Running test_openai_driver_parameter_conversion...")
test_openai_driver_parameter_conversion()

# Run the test functions from test_all_drivers
print("\n=== Running All Driver Tests ===")
import tests.drivers.openai as t_openai
import tests.drivers.gemini as t_gemini
import tests.drivers.ollama as t_ollama
import tests.drivers.huggingface as t_hf
import tests.drivers.huggingface_local as t_hfl

driver_modules = [t_openai, t_gemini, t_ollama, t_hf, t_hfl]

for mod in driver_modules:
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and name.startswith("Test"):
            print(f"Running tests in class {name}...")
            instance = obj()
            for m_name, m_obj in inspect.getmembers(instance):
                if inspect.ismethod(m_obj) and m_name.startswith("test_"):
                    print(f"  Running {m_name}...")
                    m_obj()

print("\nAll tests passed successfully!")

