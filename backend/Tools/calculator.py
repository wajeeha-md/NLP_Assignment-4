import ast
import operator
import asyncio

# Supported operators
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant): # Python 3.8+ compatibility
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError("Unsupported constant type.")
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        if type(node.op) not in _OPS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        
        # Prevent massive exponents
        if isinstance(node.op, ast.Pow):
            if right > 100:
                raise ValueError("Exponent too large.")
        
        return _OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        if type(node.op) not in _OPS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        operand = _safe_eval(node.operand)
        return _OPS[type(node.op)](operand)
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise TypeError(f"Unsupported node type: {type(node).__name__}")

def _evaluate_expression(expression: str):
    try:
        # Parse expression into an AST
        node = ast.parse(expression, mode='eval')
        return _safe_eval(node)
    except SyntaxError:
        return "Error: Invalid syntax in mathematical expression."
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {str(e)}"

async def calculate(expression: str) -> str:
    """
    Asynchronously evaluates a mathematical expression.
    
    Args:
        expression: The math expression to evaluate (e.g., "2 + 2", "10 * (5 - 3)")
        
    Returns:
        The result of the expression as a string, or an error message if invalid.
    """
    # Even though math evaluation is fast, we wrap it in a thread 
    # to stick to the async requirement and prevent any possible blocking.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _evaluate_expression, expression)
    return str(result)

if __name__ == "__main__":
    async def test():
        print("Testing valid input: '10 + 5 * 2' ->", await calculate("10 + 5 * 2"))
        print("Testing valid input: '(100 / 2) ** 2' ->", await calculate("(100 / 2) ** 2"))
        print("Testing invalid input: '10 / 0' ->", await calculate("10 / 0"))
        print("Testing invalid input: 'import os' ->", await calculate("import os"))
        print("Testing invalid syntax: '10 + * 2' ->", await calculate("10 + * 2"))

    asyncio.run(test())
