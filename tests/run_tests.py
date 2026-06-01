import sys
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
sys.path.insert(0, "/home/ahmed/projects/dingir/src")
sys.path.insert(0, "/home/ahmed/projects/dingir")

# Run the test functions
from tests.test_config_composition import (
    test_explicit_args_tracking,
    test_config_merge_rightmost_precedence,
    test_config_merge_none_ignored,
    test_invalid_config_type_raises_error,
    test_driver_composite_config,
    test_openai_driver_parameter_conversion,
    test_openai_compatible_driver,
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

print("Running test_openai_compatible_driver...")
test_openai_compatible_driver()

print("All tests passed successfully!")
