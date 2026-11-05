import importlib.metadata
import unittest

from google.genai import types
from google.genai._api_client import retry_args

from src.support_agent import _make_llm


class GeminiRetryConfigurationTests(unittest.TestCase):
    def test_installed_google_sdk_configuration_allows_one_total_attempt(self):
        self.assertEqual(
            importlib.metadata.version("langchain-google-genai"), "4.3.7"
        )
        self.assertEqual(importlib.metadata.version("google-genai"), "2.20.0")

        llm = _make_llm("unit-test-key")
        self.assertEqual(llm.max_retries, 1)

        request_config = llm._build_request_config(
            formatted_tools=None,
            formatted_tool_config=None,
            formatted_safety_settings=[],
            params=types.GenerateContentConfig(),
            cached_content=None,
            system_instruction=None,
            timeout=llm.timeout,
            max_retries=llm.max_retries,
        )
        retry_options = request_config.http_options.retry_options

        self.assertEqual(retry_options.attempts, 1)
        sdk_retry_args = retry_args(retry_options)
        self.assertEqual(sdk_retry_args["stop"].max_attempt_number, 1)


if __name__ == "__main__":
    unittest.main()
