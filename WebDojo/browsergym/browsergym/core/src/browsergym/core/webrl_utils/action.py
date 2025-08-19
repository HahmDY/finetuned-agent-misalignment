import ast

def remove_comments(webrl_response: str) -> str:
	for key in ['exit(','do(','go_backward(']:
		if key in webrl_response:
			return key + webrl_response.split(key)[-1]
	lines = webrl_response.split('\n')
	for i, line in enumerate(lines):
		if line.strip().startswith('#'):
			continue
		else:
			return '\n'.join(lines[i:])
	return ''


def parse_function_call(webrl_response):
	expression = remove_comments(webrl_response)
	expression = expression.strip() 
	tree = ast.parse(expression, mode='eval')
	func_call = tree.body
	if not isinstance(func_call, ast.Call):
		return {
			"operation": expression,
		}
	func_name = func_call.func.id
	result = {
		"operation": func_name,
	}
	args = func_call.args
	kwargs = func_call.keywords
	for kw in kwargs:
		if func_name == "do" and kw.arg == "action":
			result["action"] = ast.literal_eval(kw.value)
		else:
			if "kwargs" not in result:
				result["kwargs"] = {}
			if kw.arg == "element":
				try:
					inner_func = kw.value
					if isinstance(inner_func, ast.Call) and inner_func.func.id == 'find_element_by_instruction':
						for inner_kw in inner_func.keywords:
							if inner_kw.arg == "instruction":
								result["kwargs"]["instruction"] = ast.literal_eval(inner_kw.value)
					else:
						result["kwargs"][kw.arg] = ast.literal_eval(inner_func)
				except Exception:
					result["kwargs"][kw.arg] = ast.literal_eval(kw.value)
			else:
				result["kwargs"][kw.arg] = ast.literal_eval(kw.value)
	return result