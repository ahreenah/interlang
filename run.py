import ast

# Define a Python AST node visitor to generate JavaScript code
class JavascriptGenerator(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            args = ', '.join(ast.literal_eval(ast.Expression(body=arg)) for arg in node.args)
            print(f'console.log({args});')
        else:
            self.generic_visit(node)

# Parse a Python file and generate equivalent JavaScript code
with open('example.py') as f:
    code = f.read()
    tree = ast.parse(code)

    generator = JavascriptGenerator()
    generator.visit(tree)
