# Linear Probe Training

# Install packages
```
pip install -r requirements.txt
```

# Download datasets and train linear probe
To download each dataset:
```
cd ./data
python /download_dataset.py
```
To train linear probe:
```
python linear_probe.py --model_path meta-llama/Llama-3.1-8B-Instruct --model_tag llama
```
```model_tag``` is one of ```llama, glm, qwen```