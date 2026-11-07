"""Keep automated tests from sending observations to the configured Langfuse project."""

import os

os.environ["RESOLVEX_LANGFUSE_TRACING_ENABLED"] = "false"
