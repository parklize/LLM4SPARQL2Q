from torchmetrics.text.bert import BERTScore

import re
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4f}'.format  # Set 4 decimal places globally

def eval(res_dict: dict):
    ''' Evaluate the generated questions with ground truth '''
    bertscore = BERTScore()
    res_bertscore = bertscore(
        res_dict['generated'],
        res_dict['ground_truth'],
    )

    for i, metric in enumerate(['f1', 'precision', 'recall']):
        res_dict[f'{metric}-BERTScore'] = res_bertscore[metric].tolist()

    return res_dict


def eval_res(dir='results', model_name_or_path=None):
    ''' 
    Combine all results 
    Print out the mean and std 
    '''
    # Recompute
    #bertscore = BERTScore(
    #    model_name_or_path=model_name_or_path,
    #)

    dataframes = list()
    for file in os.listdir(dir):
        if file.endswith('.csv'):# and file.startswith('meta'):
            filepath = os.path.join(dir, file)
            df = pd.read_csv(
                filepath, 
                header=0, 
                delimiter=',', 
                quotechar='"'
            )

            # Change MISTRIAL to MISTRIALFT in results for FT
            if 'MISTRALFT' in file:
                df['template'] = df['template'].str.replace('MISTRAL', 'MISTRALFT')

            print(df)

            #res = bertscore(
            #    df['generated'].tolist(),
            #    df['ground_truth'].tolist(),
            #)
            #for metric in ['f1', 'precision', 'recall']:
            #    df[f'{metric}-BERTScore'] = res[metric].tolist()

            print(f'{file} -- df shape: {df.shape} -- {df["f1-BERTScore"].mean()}')

            if 'context' not in df.columns:
                df['context'] = False
            dataframes.append(df)

            df['query_len'] = df['sparql'].str.split().str.len()
            plt.figure(figsize=(6, 4))
            plt.scatter(df['query_len'], df['f1-BERTScore'], color='blue', marker='o', alpha=0.7)
            plt.xlabel('Query Length')
            plt.ylabel('F1')
            plt.title('Scatter Plot Example')
    
            # Save plot to a file
            plt.savefig(f'results/{file}_scatter_plot.png', dpi=300)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.reset_index(drop=True)

    print(combined_df.shape)
    print(combined_df.dtypes)
    print(combined_df.head())
    
    res = combined_df.groupby(
            ['hub_id', 'template']
    )[['precision-BERTScore', 'recall-BERTScore', 'f1-BERTScore']] \
    .agg(['mean'])
    
    print(res)
    print(res.to_latex(float_format="%.4f"))

    
    pre = combined_df[ \
        (combined_df['template'] == 'BASIC0') \
        & (combined_df['hub_id'] == 'meta-llama/Llama-2-7b-chat-hf') \
    ]['f1-BERTScore'].tolist()

    ttest(
        combined_df, 
        pre_template='BASIC0', 
        templates=['BASIC1', 'BASIC3', 'BASIC5', 'CoT0'], 
        hub_id='meta-llama/Llama-2-7b-chat-hf'
    )
    ttest(
        combined_df, 
        pre_template='LLAMA3_BASIC0', 
        templates=['LLAMA3_'+x for x in ['BASIC0', 'BASIC1', 'BASIC3', 'BASIC5', 'CoT0']], 
        hub_id='meta-llama/Meta-Llama-3-8B-Instruct'
    )
    ttest(
        combined_df, 
        pre_template='MISTRAL_BASIC0', 
        templates=['MISTRAL_'+x for x in ['BASIC0', 'BASIC1', 'BASIC3', 'BASIC5', 'CoT0']], 
        hub_id='mistralai/Mistral-7B-Instruct-v0.2'
    )
    ttest(
        combined_df, 
        pre_template='MISTRALFT_BASIC0', 
        templates=['MISTRALFT_'+x for x in ['BASIC0', 'BASIC1', 'BASIC3', 'BASIC5', 'CoT0']], 
        hub_id='mistralai/Mistral-7B-Instruct-v0.2'
    )


    combined_df['query_len'] = combined_df['sparql'].str.split().str.len()
    plt.figure(figsize=(6, 4))
    plt.scatter(combined_df['query_len'], combined_df['f1-BERTScore'], color='blue', marker='o', alpha=0.7)
    plt.xlabel('Query Length')
    plt.ylabel('F1')
    plt.title('Scatter Plot Example')
    
    # Save plot to a file
    plt.savefig('results/scatter_plot.png', dpi=300)

def ttest(df: pd.DataFrame, pre_template: str, templates: list, hub_id: str) -> None:
    """ Print ttest results 
    
    Args:
        df: Dataframe containing all results
        pre_template: pre template to filter
        templates: to filter dataframe for post treatments
        hub_id: to filter dataframe for post treatments
    """

    pre = df[ \
        (df['template'] == pre_template) \
        & (df['hub_id'] == hub_id) \
    ]['f1-BERTScore'].tolist()

    print('#'*90+' ')
    for template in templates:
        post = df[ \
            (df['template'] == template) \
            & (df['hub_id'] == hub_id) \
        ]['f1-BERTScore'].tolist()
        print(len(pre), len(post))
        print(hub_id, template, stats.ttest_rel(pre, post))


def clean_output(res: str, template: str) -> str:
    """ Clean up result to make sure the last line is question """
    res = re.sub(r'<\|.*?\|>','',res).strip()
    if template == 'MISTRAL_BASIC':
        res_final = res.split('\n')[0].replace('Question:','').strip()
    else:
        res_final = res.split('\n')[-1].replace('Question:','').strip()
        if '?' not in res_final:
            # Go through each line 
            for line in res.split('\n'):
                if '?' in line:
                    res_final = line.replace('Question:','').strip()
                    break
           
    return res_final


if __name__ == '__main__':
    eval_res(
        #model_name_or_path='bert-base-uncased'
    )
