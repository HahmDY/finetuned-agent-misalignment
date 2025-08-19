import os
import yaml
import re
import json
from web_prompt import SYSTEM_PROMPT

def process_code_data():

	source_path = "/LLaMA-Factory/data/code_raw.json"
	target_path = "/LLaMA-Factory/data/code_7k.json"

	with open(source_path, "r", encoding="utf-8") as f:
		data = [json.loads(line) for line in f]
	converted_data = []

	for obj in data:
		data_point = {}
		converted_conversations = []

		conversations = obj["conversations"]
		for conversation in conversations:
			if conversation["role"] == "system":
				data_point["system"] = conversation["content"]
			elif conversation["role"] == "user":
				converted_conversations.append({
					"from": "human",
					"value": conversation["content"]
				})
			elif conversation["role"] == "assistant":
				converted_conversations.append({
					"from": "gpt",
					"value": conversation["content"]
				})

		data_point["conversations"] = converted_conversations
		converted_data.append(data_point)

	with open(target_path, "w") as f:
		json.dump(converted_data, f)
  
def process_web_data():
    
	webrl_datapath = "/LLaMA-Factory/data/web_raw.json"
	savepath = "/LLaMA-Factory/data/web_9k.json"

	def parse_prompt_with_html(prompt: str):
		task_split = prompt.split("\n\nRound 0\n\n", 1)
		if len(task_split) != 2:
			raise ValueError("Prompt does not contain Round 0 marker properly.")

		task_instruction = task_split[0].replace("Task Instruction: ", "").strip()
		history_part = task_split[1]

		rounds = re.split(r"\n\nRound \d+\n\n", history_part)
		rounds = [r for r in rounds if r.strip()]

		action_history = []
		raw_html = None

		for i, round_text in enumerate(rounds):
			parts = re.split(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n", round_text)
			if len(parts) != 2:
				raise ValueError(f"Round {i} does not split correctly into user and assistant parts.")
			
			user_part = parts[0]
			assistant_part = parts[1]

			user_content = user_part.split("<|eot_id|><|start_header_id|>user<|end_header_id|>\n")[-1].strip()

			if raw_html is None and ("<html" in user_content or "<HTML" in user_content):
				raw_html = user_content
			
			assistant_action = assistant_part.strip()

			action_history.append(assistant_action)
		action_history = action_history[:-1]
		
		return task_instruction, action_history, raw_html


	with open(webrl_datapath, "r") as f:
		data = json.load(f)
		
	converted_data = []

	for datapoint in data:
		
		converted_datapoint = {}

		user_prompt = datapoint["conversations"][0]["value"]
		assistant_prompt = datapoint["conversations"][1]["value"]

		task_instruction, action_history, raw_html = parse_prompt_with_html(user_prompt)

		converted_conversations = []

		for i in range(len(action_history)):
			user_prompt_converted = f"Round {i}"
	
			if i == 0: 
				user_prompt_converted += f"\n\nTask Instruction: {task_instruction}"
			else:
				user_prompt_converted += f"\n\n** Simplified html **"
				
			converted_conversations.append({
				"from": "human",
				"value": user_prompt_converted
			})
			converted_conversations.append({
				"from": "gpt",
				"value": action_history[i]
			})
	
		if len(action_history) == 0:
			converted_conversations.append({
				"from": "human",
				"value": f"Round {len(action_history)}\n\nTask Instruction: {task_instruction}\n\nObservation: {raw_html}"
			})
		else:
			converted_conversations.append({
				"from": "human",
				"value": f"Round {len(action_history)}\n\nObservation: {raw_html}"
			})
		converted_conversations.append({
			"from": "gpt",
			"value": assistant_prompt
		})
			
		converted_datapoint["conversations"] = converted_conversations
		converted_datapoint["system"] = SYSTEM_PROMPT

		converted_data.append(converted_datapoint)
	

	with open(savepath, "w") as f:
		json.dump(converted_data, f, indent=4)
  
  
def process_yaml():

	file_names = [
		"web_llama",
		"web_glm",
		"web_qwen",
		"code_llama",
		"code_glm",
		"code_qwen",
	]

	for file_name in file_names:
		with open(f"./examples/train_full/{file_name}_raw.yaml", "r", encoding="utf-8") as f:
			raw_yaml = f.read()
		ping_home = os.getenv("PING_HOME")
		replaced_yaml = raw_yaml.replace("$PING_HOME", ping_home)
		with open(f"./examples/train_full/{file_name}.yaml", "w", encoding="utf-8") as f:
			f.write(replaced_yaml)

if __name__ == "__main__":
	process_code_data()
	process_web_data()
	process_yaml()