# Main Environments
* Python 3.11.0
* Others can be found in ``requirements.txt``

# Folder Structure
```python
/data				# beastiary KG files
/data_process		# data processing utils
/results			# result folder
hfpipeline.py		# HuggingFace pipeline
main.py				# entry point 
requirements.txt	# packages auto-dumped via ```pip freeze > requirements.txt```
run.sh				# bash script to run experiments
template.py			# prompt templates
utils.py			# evaluation related utils
```

# Example Usage
Using ```Llama-2-7b-chat-hf``` with BASIC template:

```$ python -m main --hub_id "meta-llama/Llama-2-7b-chat-hf" --template BASIC```


Using ```Meta-Llama-3-8B-Instruct``` with LLAMA3_BASIC template:

```$ python -m main --hub_id "meta-llama/Meta-Llama-3-8B-Instruct" --template LLAMA3_BASIC```
