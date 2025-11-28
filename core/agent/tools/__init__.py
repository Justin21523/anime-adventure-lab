"""工具函式代理，轉向 core.agents.tools.*"""

from core.agents.tools import calculator, web_search, file_ops, rag_search  # noqa: F401
from core.agents.tools.calculator import *  # noqa: F401,F403
from core.agents.tools.web_search import *  # noqa: F401,F403
from core.agents.tools.file_ops import *  # noqa: F401,F403
from core.agents.tools.rag_search import *  # noqa: F401,F403

__all__ = [
    "calculator",
    "web_search",
    "file_ops",
    "rag_search",
]
__all__ += [n for n in globals() if not n.startswith("_") and n not in __all__]
