from typing import Dict, Protocol, Any


class Tool(Protocol):
    def name(self) -> str: ...
    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name()] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


class _ToolsAdapter:
    def __init__(self, impl: Any, method: str):
        self._impl = impl
        self._method = method

    def name(self) -> str:
        return self._method

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self._impl, self._method)
        out = fn(**payload)
        return {"result": out}


def build_default_registry() -> ToolRegistry:
    from src.utils.Tools import Tools
    impl = Tools()
    registry = ToolRegistry()
    for method in dir(impl):
        if method.startswith("_"):
            continue
        if callable(getattr(impl, method)):
            registry.register(_ToolsAdapter(impl, method))
    return registry
