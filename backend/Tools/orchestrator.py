import json
import re
import asyncio
from typing import Callable, Any, Dict, List

class ToolOrchestrator:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self._result_cache: Dict[str, Any] = {}

    def register(self, tool_name: str, func: Callable, description: str = ""):
        """Register an asynchronous function as a tool with a description."""
        self._tools[tool_name] = func
        self._descriptions[tool_name] = description or func.__doc__ or "No description provided."

    def get_system_instructions(self) -> str:
        """Generate a system prompt block listing available tools and usage rules."""
        if not self._tools:
            return ""
            
        instr = (
            "TOOL CALLING POLICY:\n"
            "You have access to the following tools. Use them if relevant to the user's request:\n"
        )
        for name, desc in self._descriptions.items():
            instr += f"- {name}: {desc}\n"
            
        instr += (
            "\nTo use a tool, output ONLY a JSON block first. Do NOT include any other text in the same turn before the JSON block.\n"
            "Example:\n"
            "{\n"
            "  \"tool_name\": \"example_tool\",\n"
            "  \"arguments\": {\"arg1\": \"value1\"}\n"
            "}\n"
            "The system will execute the tool and provide the result. Then you can generate a natural response.\n"
        )
        return instr

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse raw LLM output to find tool call JSON blocks.
        Expected format:
        {
          "tool_name": "...",
          "arguments": {...}
        }
        """
        tool_calls = []
        
        start = 0
        while True:
            start = text.find('{', start)
            if start == -1:
                break
                
            open_braces = 0
            end = -1
            in_string = False
            escape = False
            
            for i in range(start, len(text)):
                c = text[i]
                if not escape and c == '"':
                    in_string = not in_string
                
                if not in_string:
                    if c == '{':
                        open_braces += 1
                    elif c == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            end = i
                            break
                            
                if c == '\\':
                    escape = not escape
                else:
                    escape = False
                    
            if end != -1:
                block = text[start:end+1]
                start = end + 1
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and "tool_name" in parsed and "arguments" in parsed:
                        if isinstance(parsed["arguments"], dict):
                            tool_calls.append(parsed)
                except json.JSONDecodeError:
                    pass
            else:
                break
                
        return tool_calls

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a single parsed tool call with caching and argument filtering."""
        import inspect
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})
        
        arg_str = json.dumps(arguments, sort_keys=True)
        cache_key = f"{tool_name}:{arg_str}"
        
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered.")
            
        func = self._tools[tool_name]
        
        # --- Safety Filter: Only pass arguments that the function accepts ---
        try:
            sig = inspect.signature(func)
            valid_args = {
                k: v for k, v in arguments.items()
                if k in sig.parameters
            }
            
            result = await func(**valid_args)
            response = {"status": "success", "result": result, "cached": False}
            self._result_cache[cache_key] = {**response, "cached": True}
            return response
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def execute_all(self, text: str) -> List[Dict[str, Any]]:
        """Parse and execute all tool calls found in the text."""
        calls = self.parse_tool_calls(text)
        results = []
        for call in calls:
            res = await self.execute_tool(call)
            results.append({
                "tool_call": call,
                "execution": res
            })
        return results

# Default global instance
orchestrator = ToolOrchestrator()

if __name__ == "__main__":
    # --- Standalone Test ---
    async def dummy_weather_tool(location: str, unit: str = "C"):
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Weather in {location} is 25 {unit}"
        
    async def dummy_crm_tool(user_id: str):
        await asyncio.sleep(0.1)
        return {"name": "Test User", "budget": "1 Crore"}

    orchestrator.register("get_weather", dummy_weather_tool)
    orchestrator.register("get_crm_info", dummy_crm_tool)

    raw_llm_output = """
    I can certainly help you with that. Let me check the weather first.
    {
      "tool_name": "get_weather",
      "arguments": {"location": "Lahore"}
    }
    
    And let me also check your user profile!
    {
      "tool_name": "get_crm_info",
      "arguments": {"user_id": "u123"}
    }
    """

    async def run_tests():
        print("--- Testing Tool Orchestrator ---")
        print("Parsing LLM Output...")
        calls = orchestrator.parse_tool_calls(raw_llm_output)
        print(f"Found {len(calls)} valid tool calls.")
        for c in calls:
            print(f" -> {c}")
            
        print("\nExecuting Tool Calls...")
        results = await orchestrator.execute_all(raw_llm_output)
        for r in results:
            print(f"Result for '{r['tool_call']['tool_name']}': {r['execution']}")
            
        print("--- Test Complete ---")

    asyncio.run(run_tests())
