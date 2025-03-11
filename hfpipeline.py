from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from template import TEMPLATES
from transformers import logging, MistralForCausalLM

import torch
import transformers

logging.set_verbosity_error()

transformers.set_seed(1000)
torch.manual_seed(1000)

class HFPipeline:
    def __init__(self, 
        hub_id: str,
        template: str,
        adapter: str = None
    ):
        ''' Initialize HFPipeline 
        
        Args:
            hub_id: the HuggingFace hub model id to download the model
            template: prompt template to be used for inference

        '''
        self.hub_id = hub_id
        self.adapter = adapter
        if template in TEMPLATES:
            self.template = TEMPLATES[template]
        else:
            logging.info('Not defined template')
        self.load()
        self.pipeline()

    def load(self,):
        ''' Load the model from HuggingFace '''
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hub_id,
        )
        
        model_config = transformers.AutoConfig.from_pretrained(
            self.hub_id,
            top_k=10,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            repeatition_penalty=1.1,
            return_full_text=False,
        )
        
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(
        #self.llm = transformers.MistralForCausalLM.from_pretrained(
            self.hub_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            torch_dtype='auto',
            device_map='auto',
        )
        
        if self.adapter:
            self.llm.load_adapter(self.adapter)
        
        self.llm.eval()
        torch.no_grad()
        

    def pipeline(self,):
        ''' Load the pipeline '''
        pipeline = transformers.pipeline(
            model=self.llm, 
            tokenizer=self.tokenizer, 
            task="text-generation",
            max_new_tokens=1000,
        )
        
        hf = HuggingFacePipeline(
            pipeline=pipeline,
        )

        prompt = PromptTemplate(
            input_variables=['nshots', 'sparql', 'context'],
            template=self.template
        )

        self.pipeline = prompt | hf 

        
    
