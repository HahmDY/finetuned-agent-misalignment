export DATASET=webarena

# Below are the variables you should set for the evaluation.
export SHOPPING="your_url:7770"
export REDDIT="your_url:9999"
export SHOPPING_ADMIN="your_url:7780/admin"
export GITLAB="your_url:8023"
export MAP="your_url:3000"
export WIKIPEDIA="your_url:8888"
export HOMEPAGE="your_url:4399"


export OPENAI_API_KEY=your_key


export OPENAI_API_URL=https://api.openai.com/v1

# Optional: you can set the following variables to evaluate the preset model in llms/providers/api_utils.py
export GEMENI_API_KEY=your_key
export QWEN_API_KEY=your_key
export CLAUDE_API_KEY=your_key

# Optional: if you have trained your model, we recommend deploying it as an API service, where you can set a FINETUNED_URL to evaluate it.
export FINETUNED_URL=http://localhost:8000/v1
