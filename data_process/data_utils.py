import re
import json
import pprint
import logging

from rdflib import Graph, Namespace

logger = logging.getLogger(__name__)


def load_data(json_file: str):
    """ Load JSON formatted data according to QALD format 

    Args:
        json_file: the json file path

    Returns:
        Loaded JSON

    Raises:
        FileNotFoundError: when file is not found
        JSONDecoderError: when JSON format has issues
        Exception: other unexpected
    """
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f'The file {json_file} was not found')
    except json.JSONDecodeError as e:
        logger.error(f'Failed to decode JSON file: {json_file}')
    except Exception as e:
        logger.error(f'An unexpected error occured while loading: {json_file}')


def shorten_uri(g: Graph, uri: str) -> str:
    """ Return a shorten URI """
    if uri.startswith('http'):
        return g.qname(uri)
    else:
        return uri


def get_context(g: Graph, uri: str) -> list:
    """ 
    If it is property: get all ontology information? rdf:type and domain and range
    If it is entity: get rdf:type

    Args:
        g: Graph from rdflib
        uri: target URI to get the context

    Returns:
        list of triples ()
    """
    q = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?type ?domain ?range WHERE {{ 
            {uri} rdf:type ?type .
            OPTIONAL {{ {uri} rdfs:domain ?domain }}
            OPTIONAL {{ {uri} rdfs:range ?range }}
        }}
    """
    res = list()
    qres = g.query(q)
    for row in qres:
        uri = shorten_uri(g, uri.replace('<','').replace('>',''))
        uri_type = str(row.type)
        res.append(f"({uri}, rdf:type, {shorten_uri(g, uri_type)})")
        if row.domain:
            uri_domain = str(row.domain)
            res.append(f'({uri}, rdfs:domain, {shorten_uri(g, uri_domain)})')
        if row.range:
            uri_range = str(row.range)
            res.append(f'({uri}, rdfs:range, {shorten_uri(g, uri_range)})')

    return list(set(res))


def get_query_context(g: Graph, query: str) -> str:
    """
    Process query to extract uris, 
    query to get context list and return as str

    Args:
        g: Graph in rdflib to query
        query: SPARQL query extended

    Returns:
        context as str with one triple per line
    """
    pattern = r'(<[^<>]+>)'
    uris = re.findall(pattern, query)
    uris = list(set(uris))
    logger.debug(f'URIs in query: {uris}')
    
    triple_list = list()
    for uri in uris:
        triple_list += get_context(g, uri)

    context = '\n'.join(triple_list)
   
    return context


def shorten_sparql_uris(g: Graph, query: str) -> str:
    """ Return shorten SPARQL query """
    pattern = r'<([^<>]+)>'

    def replace_uri(match):
        full_uri = match.group(1)
        try:
            return g.qname(full_uri)
        except Exception:
            return f'<{full_uri}>'

    return re.sub(pattern, replace_uri, query)


def load_graph(kg_file: str):
    """ Load and return Graph """

    g = Graph()
    g.parse(kg_file)

    g.bind('rdf', Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'))
    g.bind('rdfs', Namespace('http://www.w3.org/2000/01/rdf-schema#'))
    g.bind('beasiary', Namespace('http://www.semanticweb.org/annab/ontologies/2022/3/ontology#'))
    g.bind('owl', Namespace('http://www.w3.org/2002/07/owl#'))

    return g


if __name__ == '__main__':
    g = load_graph('data/beastiary_kg.rdf')
    #get_context(g,'<http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasResists>')
    get_context(g,'<http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages>')
