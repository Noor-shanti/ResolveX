import sys
import types
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from src.llm_provider import configured_provider, create_chat_model
from src.support_agent import make_m5_chains


class LLMProviderTests(unittest.TestCase):
    def test_provider_selection_is_explicit_and_gemini_remains_default(self):
        self.assertEqual(configured_provider({}), "gemini")
        self.assertEqual(configured_provider({"LLM_PROVIDER": "GEMINI"}), "gemini")
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            configured_provider({"LLM_PROVIDER": "automatic"})
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            configured_provider({"LLM_PROVIDER": "local"})

    def test_gemini_selection_preserves_existing_model_configuration(self):
        captured = {}

        class FakeGemini:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = types.ModuleType("langchain_google_genai")
        module.ChatGoogleGenerativeAI = FakeGemini
        with patch.dict(sys.modules, {"langchain_google_genai": module}):
            model = create_chat_model(api_key="test-key", provider="gemini", environ={})

        self.assertIsInstance(model, FakeGemini)
        self.assertEqual(captured["model"], "gemini-3.5-flash-lite")
        self.assertEqual(captured["max_retries"], 1)
        self.assertEqual(captured["timeout"], 60)

    def test_single_structured_chain_accepts_the_selected_model(self):
        calls = []

        class FakeModel:
            def with_structured_output(self, schema, **kwargs):
                calls.append((schema, kwargs))
                return RunnableLambda(lambda _input: {})

        with patch("src.support_agent._make_llm", return_value=FakeModel()):
            chains = make_m5_chains(provider="gemini")

        self.assertEqual(set(chains), {
            "chain", "action_extractor", "evidence_assessor", "conflict_assessor",
        })
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1].get("method"), "json_schema")
        self.assertEqual(calls[0][1].get("include_raw"), True)
        self.assertIsNone(chains["action_extractor"])
        self.assertIsNone(chains["evidence_assessor"])
        self.assertIsNone(chains["conflict_assessor"])

if __name__ == "__main__":
    unittest.main()
