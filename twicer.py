''' node.name = '__twice'
            node.body = node.body * 2
        return node

def twice():
    pass

def _print(*args, **kwargs):
    return builtins.print(*args, **kwargs)

src = '''
twice:
    _print('hello')
'''



src = """
twice:
    print('hello')
"""

# Parse the source code into an abstract syntax tree
ast_obj = ast.parse(src)

# Transform the syntax tree to replace `twice:` blocks with `for` loops
twice_transformer = TwiceTransformer()
new_ast_obj = twice_transformer.visit(ast_obj)

# Compile the modified syntax tree into a code object
code_obj = compile(new_ast_obj, filename='<string>', mode='exec')

# Execute the code object
exec(code_obj)

#ast_obj = ast.parse(src)
#ast_obj = TwiceTransformer().visit(ast_obj)
#code_obj = compile(ast_obj, filename="<ast>", mode="exec")
# exec(code_obj, {'twice': twice, '_print': _print})
'''



import ast

class TwiceTransformer(ast.NodeTransformer):
    def visit_TwiceBlock(self, node):
        # Replace `twice:` block with a for loop that executes the body twice
        return ast.For(target=ast.Name(id='_', ctx=ast.Store()),
                       iter=ast.Range(lower=ast.Constant(value=2), upper=ast.Constant(value=3)),
                       body=node.body,
                       orelse=[])

src = """
twice:
    print('hello')
"""

# Preprocess the source code to replace `twice:` blocks with `_` loops
src = src.replace('twice:', '_ in range(2):')

# Parse the source code into an abstract syntax tree
ast_obj = ast.parse(src)

# Transform the syntax tree to replace `_` loops with `for` loops
twice_transformer = TwiceTransformer()
new_ast_obj = twice_transformer.visit(ast_obj)

# Compile the modified syntax tree into a code object
code_obj = compile(new_ast_obj, filename='<string>', mode='exec')

# Execute the code object
exec(code_obj)
