# ADR 0003: Isolate the public demo

Status: Accepted

The public portfolio must not expose model control, filesystem tools, private
world data, or persistent mutation endpoints. It is deployed as a static site
with a small deterministic API that never imports the core application.

The complete FastAPI/React stack is a separate authenticated deployment. This
prevents a mock-mode or feature-flag mistake from exposing the private API.
