START = """[INST]<<SYS>>
You are SPARQL (SPARQL Protocol and RDF Query Language) expert.
Your goal is to analyze and interpret a given SPARQL query and provide the corresponding
question to lay users who do not have any SPARQL knowledge and have no access to the query.
<</SYS>>

Given a SPARQL query, provide the corresponding question at the end.
Your answer should be self-explanatory and should contain only the corresponding question without any other additional text.
"""

BODY = """
{nshots}
{context}

The SPARQL query is between [SPARQL] and [/SPARQL] tags as below:
[SPARQL]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX beasiary: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

{sparql}
[/SPARQL]
"""

END = """

The corresponding question to the SPARQL query:
[/INST]
"""

BASIC = START + BODY + END
MISTRAL_BASIC = START.replace('<<SYS>>','').replace('<</SYS>>','[/INST]\n[INST]') + BODY + END

TSTART = """[INST]<<SYS>>
You are SPARQL (SPARQL Protocol and RDF Query Language) expert.
Your goal is to analyze and interpret a given SPARQL query and provide the corresponding
question to lay users who do not have any SPARQL knowledge and have no access to the query.
<</SYS>>

Given a SPARQL query, provide the corresponding natural language question at the end.
"""

TBODY = """
{nshots}
{context}

The SPARQL query is between [SPARQL] and [/SPARQL] tags as below:
[SPARQL]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX beasiary: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

{sparql}
[/SPARQL]
"""

FT = """

Transform the SPARQL query to a natural language question.
"""

TEND = """

The last line of your output should be in the following format containing the corresponding question ONLY WITHOUT any additional text such as explanation and further questions:

Generate a response strictly in the following format. Do not include any extra words, explanations, or closing remarks.
Format:
Reasoning: <your reasoning steps or thoughts>
Question: <the corresponding natural language question of the given SPARQL query>

Take a deep breath and work on this problem step by step.
[/INST]
"""

CoT = TSTART + TBODY + TEND
MISTRAL_CoT = TSTART.replace('<<SYS>>','').replace('<</SYS>>','[/INST]\n[INST]') + TBODY + FT + TEND

LLAMA3_START = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are SPARQL (SPARQL Protocol and RDF Query Language) expert.
Your goal is to analyze and interpret a given SPARQL query and provide the corresponding
question to lay users who do not have any SPARQL knowledge and have no access to the query.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Given a SPARQL query, provide the corresponding question at the end.
Your answer should be self-explanatory and should contain only the corresponding question without any other additional text.
"""

LLAMA3_BODY = """
{nshots}
{context}

The SPARQL query is between [SPARQL] and [/SPARQL] tags as below:
[SPARQL]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX beasiary: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

{sparql}
[/SPARQL]
"""

LLAMA3_END = """

The corresponding question to the SPARQL query:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

LLAMA3_BASIC = LLAMA3_START + LLAMA3_BODY + LLAMA3_END

LLAMA3_TSTART = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are SPARQL (SPARQL Protocol and RDF Query Language) expert.
Your goal is to analyze and interpret a given SPARQL query and provide the corresponding
question to lay users who do not have any SPARQL knowledge and have no access to the query.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Given a SPARQL query, provide the corresponding natural language question at the end.
"""

LLAMA3_TBODY = """
{nshots}
{context}

The SPARQL query is between [SPARQL] and [/SPARQL] tags as below:
[SPARQL]
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX beasiary: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

{sparql}
[/SPARQL]
"""

LLAMA3_TEND = """

The last line of your output should be in the following format containing the corresponding question ONLY WITHOUT any additional text such as explanation and further questions:

Generate a response strictly in the following format. Do not include any extra words, explanations, or closing remarks.
Format:
Reasoning: <your reasoning steps or thoughts>
Question: <the corresponding natural language question of the given SPARQL query>

Take a deep breath and work on this problem step by step.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
[/INST]
"""

LLAMA3_CoT = LLAMA3_TSTART + LLAMA3_TBODY + LLAMA3_TEND


TEMPLATES = {
    'BASIC': BASIC,
    'CoT': CoT,
    'LLAMA3_CoT': LLAMA3_CoT,
    'LLAMA3_BASIC': LLAMA3_BASIC,
    'MISTRAL_BASIC': MISTRAL_BASIC,
    'MISTRAL_CoT': MISTRAL_CoT
}
