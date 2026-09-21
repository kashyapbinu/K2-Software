"""
AI assistant layer: provider-agnostic LLM client, rocket-state context
builder, tool dispatch, and the chat session loop.

Nothing in here imports Qt except ``ai.tools`` (main-thread bridge), so the
provider + session can be exercised headless from the terminal.
"""
