from vllm import LLM, SamplingParams
import torch, ray
import gc
import threading
import time
import queue
from typing import Dict, Any, List, Optional

from webdojo.agent.prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SAFE

SYSTEM_PROMPT = SYSTEM_PROMPT

class WebRL_Agent_VLLM:
	def __init__(self, model_name: str, lora_path: str = None, system_prompt: str = "default"):
		"""
		Initialize the WebRL_Agent_VLLM with vLLM for faster inference.
		
		Args:
			model_name: The name or path of the model to load
			lora_path: Path to LoRA adapter (Note: current vLLM version may have limited LoRA support)
			system_prompt: "default" or "safe" to use the corresponding system prompt
		"""
		# Clear CUDA cache before initialization
		torch.cuda.empty_cache()
		gc.collect()

		# Initialize the vLLM model - automatically handles tokenizer and model together
		self.model = LLM(
			model=model_name,
			tensor_parallel_size=torch.cuda.device_count(),  # Adjust based on available GPUs
			dtype="bfloat16" if "gemma" in model_name.lower() else "half",  # Use bfloat16 for gemma, else half precision (FP16)
			trust_remote_code=True if "glm" in model_name else False,
			gpu_memory_utilization=0.7,  # Reduce from 0.9 to prevent OOM errors
			max_model_len=16384 # original 16384
		)
		
		# Set system prompt
		if system_prompt == "safe":
			self.system_prompt = SYSTEM_PROMPT_SAFE
		else:
			self.system_prompt = SYSTEM_PROMPT
		
		# Note on LoRA: vLLM support for LoRA may be limited; verify compatibility
		if lora_path:
			print(f"Warning: LoRA support in vLLM may be limited. Trying to load {lora_path}")
			# The usage of LoRA with vLLM may require specific implementation
			# Consult vLLM documentation for the latest LoRA support

		self.goal = None
		self.action_history = []
  
  
	def reset_agent(self, new_goal):
		"""Reset the agent state with a new goal."""
		self.goal = new_goal
		self.action_history = []
		self.user_prompt = None
		
		# Clear CUDA cache when resetting agent
		torch.cuda.empty_cache()
		gc.collect()
  
	def _generate_with_timeout(self, prompt, sampling_params, timeout=30):
		"""
		Generate a response with a timeout.
		
		Args:
			prompt: The input prompt
			sampling_params: Sampling parameters for vLLM
			timeout: Timeout in seconds (default: 30s)
			
		Returns:
			Generated text or None if timeout occurs
		"""
		result_queue = queue.Queue()
		
		def generate_fn():
			try:
				outputs = self.model.generate(
					prompts=[prompt],
					sampling_params=sampling_params
				)
				result_queue.put(outputs)
			except Exception as e:
				result_queue.put(e)
		
		# Start generation in a separate thread
		thread = threading.Thread(target=generate_fn)
		thread.daemon = True  # Set as daemon so it gets killed if the main thread exits
		thread.start()
		
		# Wait for the thread to complete or timeout
		thread.join(timeout)
		
		if thread.is_alive():
			# Thread is still running after timeout
			print(f"Generation timed out after {timeout} seconds")
			# We can't kill the thread or stop VLLM mid-generation
			# But we can return a timeout result
			return None
		
		# Get result from queue
		if not result_queue.empty():
			result = result_queue.get()
			if isinstance(result, Exception):
				raise result
			return result
		
		return None

	def get_response(self, html: str, prefix: str = "", response_timeout: int = 30) -> Dict[str, str]:
		"""
		Get a response from the model using vLLM for faster inference.
		
		Args:
			html: The HTML content to process
			prefix: Optional prefix to prepend to the model's response
			response_timeout: Timeout in seconds for the response generation
			
		Returns:
			Dictionary containing the model output and user prompt
		"""
		# Set system prompt
		prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
  
		index = len(self.action_history)
		history = ""
		for i in range(index - 1, -1, -1):
			if i == 0:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{self.goal}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
			else:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n** Simplified html **\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
		if len(history) + len(html) > (16384 - 512):
			truncated_length = (16384 - 512) - len(history)
			print(f"HTML too long, truncating from {len(html)} to {truncated_length} characters")
			html = html[:truncated_length]
		current_turn = f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{html}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
  
		if prefix:
			current_turn = current_turn + prefix
  
		user_prompt = f"Task Instruction: {self.goal}\n\n{history}{current_turn}"
		self.user_prompt = user_prompt
  
		prompt = prompt.format(SYSTEM_PROMPT=self.system_prompt) + user_prompt
   
		# Clear CUDA cache before generation
		torch.cuda.empty_cache()
		gc.collect()
		
		# Define sampling parameters with lower token limits
		sampling_params = SamplingParams(
			temperature=0.0,  # Equivalent to do_sample=False
			max_tokens=512,
			stop_token_ids=[],  # Add stop token IDs if necessary
			stop="<|eot_id|>"   # Using the stop token as string
		)
  
		try:
			# Generate response using vLLM with timeout
			print(f"Generating response with {response_timeout}s timeout...")
			start_time = time.time()
			
			outputs = self._generate_with_timeout(
				prompt=prompt,
				sampling_params=sampling_params,
				timeout=response_timeout
			)
			
			generation_time = time.time() - start_time
			print(f"Generation completed in {generation_time:.2f} seconds")
			
			if outputs is None:
				# Timeout occurred
				print(f"Response generation timed out after {response_timeout} seconds")
				empty_response = "The model response timed out. Please try again."
				self.action_history.append(empty_response)
				
				# Aggressively clear memory
				torch.cuda.empty_cache()
				gc.collect()
				
				return {
					"model_output": empty_response,
					"user_prompt": self.user_prompt,
				}
			
			# Extract the generated text (vLLM handles tokenization and decoding)
			generated_text = outputs[0].outputs[0].text
			
			# Clear outputs immediately
			del outputs
			torch.cuda.empty_cache()
	  
			# Apply prefix if specified (although it's already included in the prompt)
			model_output = generated_text
	  
			self.action_history.append(model_output)
	  
			print("Raw response:", model_output)
			
			return {
				"model_output": model_output,
				"user_prompt": self.user_prompt,
			}
			
		except Exception as e:
			print(f"Error generating response: {e}")
			torch.cuda.empty_cache()
			gc.collect()
			
			# Return error response
			error_response = f"Error during response generation: {str(e)[:100]}..."
			self.action_history.append(error_response)
			
			return {
				"model_output": error_response,
				"user_prompt": self.user_prompt,
			}

class WebRL_Agent_VLLM_Suffix(WebRL_Agent_VLLM):
	def get_response(self, html: str, prefix: str = "", response_timeout: int = 30) -> Dict[str, str]:
		"""
		Get a response from the model using vLLM for faster inference.
		
		Args:
			html: The HTML content to process
			prefix: Optional prefix to prepend to the model's response
			response_timeout: Timeout in seconds for the response generation
			
		Returns:
			Dictionary containing the model output and user prompt
		"""
		# Set system prompt
		prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
  
		index = len(self.action_history)
		history = ""
		for i in range(index - 1, -1, -1):
			if i == 0:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{self.goal}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
			else:
				history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n** Simplified html **\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{self.action_history[i]}\n\n" + history
		if len(history) + len(html) > (16384 - 512):
			truncated_length = (16384 - 512) - len(history)
			print(f"HTML too long, truncating from {len(html)} to {truncated_length} characters")
			html = html[:truncated_length]
		current_turn = f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{html}\n\n{prefix}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
  
		user_prompt = f"Task Instruction: {self.goal}\n\n{history}{current_turn}"
		self.user_prompt = user_prompt
  
		prompt = prompt.format(SYSTEM_PROMPT=self.system_prompt) + user_prompt
   
		# Clear CUDA cache before generation
		torch.cuda.empty_cache()
		gc.collect()
		
		# Define sampling parameters with lower token limits
		sampling_params = SamplingParams(
			temperature=0.0,  # Equivalent to do_sample=False
			max_tokens=512,
			stop_token_ids=[],  # Add stop token IDs if necessary
			stop="<|eot_id|>"   # Using the stop token as string
		)
  
		try:
			# Generate response using vLLM with timeout
			print(f"Generating response with {response_timeout}s timeout...")
			start_time = time.time()
			
			outputs = self._generate_with_timeout(
				prompt=prompt,
				sampling_params=sampling_params,
				timeout=response_timeout
			)
			
			generation_time = time.time() - start_time
			print(f"Generation completed in {generation_time:.2f} seconds")
			
			if outputs is None:
				# Timeout occurred
				print(f"Response generation timed out after {response_timeout} seconds")
				empty_response = "The model response timed out. Please try again."
				self.action_history.append(empty_response)
				
				# Aggressively clear memory
				torch.cuda.empty_cache()
				gc.collect()
				
				return {
					"model_output": empty_response,
					"user_prompt": self.user_prompt,
				}
			
			# Extract the generated text (vLLM handles tokenization and decoding)
			generated_text = outputs[0].outputs[0].text
			
			# Clear outputs immediately
			del outputs
			torch.cuda.empty_cache()
	  
			# Apply prefix if specified (although it's already included in the prompt)
			model_output = generated_text
	  
			self.action_history.append(model_output)
	  
			print("Raw response:", model_output)
			
			return {
				"model_output": model_output,
				"user_prompt": self.user_prompt,
			}
			
		except Exception as e:
			print(f"Error generating response: {e}")
			torch.cuda.empty_cache()
			gc.collect()
			
			# Return error response
			error_response = f"Error during response generation: {str(e)[:100]}..."
			self.action_history.append(error_response)
			
			return {
				"model_output": error_response,
				"user_prompt": self.user_prompt,
			}
