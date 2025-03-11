#!/bin/bash

models=(
    #"meta-llama/Llama-2-7b-chat-hf"
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    #"perevalov/Mistral-7B-instruct-v0.2-SPARQL2NL"
)

for m in "${models[@]}";
do
    echo "Cleaning up HuggingFace Hub cache..."
    #rm -r ~/.cache/huggingface/hub/*-Llama*
    #rm -r ~/.cache/huggingface/hub/*-mistralai*

    #for i in 0 1 3 5; 
    #do
    #    echo "Running with $m $i-shots"
    #    if [[ "$m" == "meta-llama/Llama-2-7b-chat-hf" ]];
    #    then
    #        python -m main --hub_id $m -n $i
    #    elif [[ "$m" == *"Mistral-7B-Instruct-v0.2"* ]];
    #    then
    #        python -m main --hub_id $m -n $i --template MISTRAL_BASIC
    #    else
    #        python -m main --hub_id $m -n $i --template LLAMA3_BASIC
    #    fi
    #done
    # Run CoT
    if [ "$m" == "meta-llama/Llama-2-7b-chat-hf" ];
    then
        python -m main --hub_id $m --template CoT
    elif [[ "$m" == *"Mistral-7B-Instruct-v0.2"* ]];
    then
        #python -m main --hub_id $m --template MISTRAL_CoT
	python -m main --hub_id $m --template MISTRAL_CoT -a 'perevalov/Mistral-7B-instruct-v0.2-SPARQL2NL'
    else
        python -m main --hub_id $m --template LLAMA3_CoT
    fi
done

