# English-only comments
from abc import ABC, abstractmethod
class LLMAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str: ...
