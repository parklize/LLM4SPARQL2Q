from data_process import data_utils
from hfpipeline import HFPipeline
from template import BASIC, LLAMA3_BASIC
from transformers import logging
from collections import defaultdict

logging.set_verbosity_error()

import time
import transformers
import torch
import argparse
import logging
import utils
import pandas as pd


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


logger = logging.getLogger('main_logger')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--json_file',
        type=str,
        help='Path to the QALD format JSON file',
        default='data/beastiary_with_qald_format.json'
    )
    parser.add_argument(
        '--hub_id',
        type=str,
        help='HuggingFaceHub model id. Default: meta-llama/Llama-2-7b-chat-hf',
        default='meta-llama/Llama-2-7b-chat-hf'
    )
    parser.add_argument(
        '--template',
        type=str,
        help='options: [BASIC,]',
        default='BASIC'
    )
    parser.add_argument(
        '--example_idx',
        nargs='+',
        type=int,
        help='Examples to be used for n-shots prompting, exclude for teesting',
        default=[23,73,7,49,91]
    )
    parser.add_argument(
        '-n', '--nshots',
        type=int,
        help='n-shot example num, up to 5',
        default=0
    )
    parser.add_argument(
        '-g', '--rdf_file',
        type=str,
        help='Path to the RDF file to load with rdflib',
        default='data/beastiary_kg.rdf'
    )
    parser.add_argument(
        '--add_context',
        action='store_true',
        help='Enable context from KG'
    )
    parser.add_argument(
        '-a', '--adapter',
        type=str,
        help='Add adapter used for fine-tuning the current model',
        default=None
    )

    args = parser.parse_args()
    logger.info(f'Args: {args}')

    ## Load KG
    g = data_utils.load_graph(args.rdf_file)

    ## Load the dataset (questions, sparql)
    q_list = data_utils.load_data(args.json_file)['questions']
    logger.info(f'Dataset size: {len(q_list)}')

    ## Initialize HFpipeline
    hf = HFPipeline(args.hub_id, args.template, args.adapter)

    ## Initialize context empty
    context = ''
    if args.add_context:
        context += '\nSome useful information in (subject, property, object) format about URIs mentioned in the SPARQL is as follows:\n'

    ## Prepare n-shots
    nshots = ''
    if args.nshots > 0:
        for shot_idx in range(args.nshots):
            nshots += '\n'
            q = q_list[args.example_idx[shot_idx]]

            if args.add_context:
                example_context = context
                example_context += data_utils.get_query_context(g, q['query']['sparql']) 
                nshots += example_context

            nshots += f"""

            The SPARQL query is between [SPARQL] and [/SPARQL] tags as below:
            [SPARQL]
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX beasiary: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

            {data_utils.shorten_sparql_uris(g, q["query"]["sparql"])}
            [/SPARQL]
            """
            
            nshots += f"""

            The corresponding question to the SPARQL query:
            {q["question"][0]["string"]}
            """
        nshots += '\n'

    logging.debug(f'nshots: {nshots}')

    ## Loop over to do the inference
    result_dict = defaultdict(list)
    for i, q in enumerate(q_list):
        if i not in args.example_idx:
            logging.debug('#'*90)
            logging.debug(f"Q{i+1}: {data_utils.shorten_sparql_uris(g, q['query']['sparql'])}")
            logging.debug(f'Ground truth: {q["question"][0]["string"]}')

            sparql = q['query']['sparql']
            result_dict['sparql'].append(sparql)
            result_dict['ground_truth'].append(q['question'][0]['string'])

            if args.add_context:
                context += data_utils.get_query_context(g, sparql) 
                #logger.debug(f'context: {context}')

            res = hf.pipeline.invoke({
                'nshots': nshots,
                'sparql': data_utils.shorten_sparql_uris(g, q['query']['sparql']),
                'context': context
            })
            logging.debug(f'Generated: {res}')

            res = utils.clean_output(res, args.template)
            logging.debug(f'Generated striped: {res}')

            result_dict['generated'].append(res)
            result_dict['template'].append(args.template+str(args.nshots))
            result_dict['hub_id'].append(args.hub_id)
            result_dict['context'].append(args.add_context)

            torch.cuda.empty_cache()

            #if i > 2:
            #    break
        else:
            logger.info(f'Bypass the example in n-shot')
    
    ## Evaluate & save the results in .csv file
    result_dict = utils.eval(result_dict)
    if args.adapter:
        filename = f'results/{args.hub_id.replace("/", "-")}_{args.template}_FT_{args.nshots}.csv'
    else:
        filename = f'results/{args.hub_id.replace("/", "-")}_{args.template}_{args.nshots}.csv'

    logger.info(f'Saving results into {filename}')

    pd.DataFrame(result_dict).to_csv(
        filename,
        index=False
    )
    


if __name__ == '__main__':
    stime = time.time()
    main()
    etime = time.time()
    logging.info(f'Elapsed time: {etime-stime} seconds')
