# ADR 0001: Story-first product boundary

Status: Accepted

The repository previously presented Story, RAG, Agent, T2I, VLM, training,
export, and post-processing as equally complete products. That breadth hid the
only reviewer-facing workflow and made readiness claims difficult to verify.

The supported product is now the Story workbench. World, knowledge retrieval,
reviewable Agent decisions, and scene generation must serve a Story turn.
Other AI capabilities remain experimental until they have their own verified
runtime, tests, and user journey.
