import get_esi_prediction
import json

available_functions = {
    "get_esi_prediction": get_esi_prediction,
}

def execute_tool_call(tool_call):
    """Parse and execute a single tool call"""
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    
    return function_to_call(**function_args)