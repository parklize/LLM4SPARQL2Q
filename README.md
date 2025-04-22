# Abstract
SPARQL queries play a crucial role in exploring knowledge graphs (KGs) and have been widely used in practice. However, understanding what questions are actually asked to KGs by exploring queries directly is a daunting task. In line with recent efforts to leverage Large
Language Models (LLMs) for deriving underlying questions of SPARQL queries, we further investigate whether increasing the number of examples in prompting and Chain-of-Thought prompting can improve the performance. Additionally, we examine whether a fine-tuned LLM with one dataset can be used on another dataset to further improve performance.

# Main Environments
* Python 3.11.11
* Others can be found in ``requirements.txt``

# Folder Structure
```python
📁data              # beastiary KG files
📁data_process      # data processing utils
📁results           # result folder
📄hfpipeline.py     # HuggingFace pipeline
📄main.py           # entry point 
📄requirements.txt  # packages auto-dumped via ```pip freeze > requirements.txt```
📄run.sh            # bash script to run experiments
📄template.py       # prompt templates
📄utils.py          # evaluation related utils
```

# Example Usage
Using ```Llama-2-7b-chat-hf``` with BASIC template:

```bash
$ python -m main --hub_id "meta-llama/Llama-2-7b-chat-hf" --template BASIC
```


Using ```Meta-Llama-3-8B-Instruct``` with LLAMA3_BASIC template:

```bash
$ python -m main --hub_id "meta-llama/Meta-Llama-3-8B-Instruct" --template LLAMA3_BASIC
```

# Citation
Guangyuan Piao, Pournima Sonawane, Shraddha Gupta, and Aidan OMahony. "Exploring the Underlying Questions of SPARQL Queries with LLMs". ESWC'25 Poster [[PDF]](https://parklize.github.io/publications/ESWC2025.pdf) [[BibTex](eswc25.bib)]
