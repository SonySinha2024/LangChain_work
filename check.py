'''
azure_invoker.py
This module defines the AzureInvoker class, responsible for invoking Azure Language Models. It handles model configuration, builds the LLM object, and manages streaming of responses, including retries and trace updates. The class is designed for seamless integration with Azure OpenAI services and efficient result processing.
Copyright © 2024
'''

import json
import traceback
import jsonpickle
import asyncio
from langchain_openai import AzureChatOpenAI
from datetime import datetime
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.callbacks import AsyncIteratorCallbackHandler
from model_apps.lib.utilities.smart_utils import logger, timer_func
from model_apps.smart_llm_loaders.prompt_loaders import (
    create_chat_prompt,create_chat_prompt_for_images
)
from model_apps.lib.utilities.smart_utils import logger
from model_apps.smart_llm_loaders.smart_model_invoker import SmartPalModelInvoker
from model_apps.smart_llm_loaders.exceptions import InvalidLLMModelError, GenericError

from model_apps.smart_llm_loaders.config_classes import ModelConfigLoader
from model_apps.smart_llm_loaders.config_loader import ConfigLoader
from model_apps.services.smart_llm_orchestrator import update_trace_url,langfuse_input_output_cost_update
from model_apps.smart_llm_loaders.search_utils import restructure_retriever_results

RETRY_COUNT = 0

class AzureInvoker(SmartPalModelInvoker):
    '''
    Invokes the Azure Language Models based on the provided Request object.
    '''

    def __init__(self, model_name):
        '''
        Initialize the object with the model name.
        '''
        self.model_name = model_name
        self.config_env = ModelConfigLoader()
        self.config_env.add_model_config("azure", model_name)
        
        # Call method to set up the configuration and LLM
        self.setup_configuration()

    def setup_configuration(self):
        """
        Sets up the configuration and LLM settings based on the model name.
        """
        # Fetch configuration dynamically
        fetcher = ConfigLoader(self.config_env, "azure", self.model_name, "chatgpt")
        config = fetcher.get_generation_chain_config()
        
        if config:
            self.deployment_name = config['deployment_name']
            self.base_url = config['endpoint_url']
            self.api_key = config['api_key']
            self.api_version = config['api_version']
        else:
            raise InvalidLLMModelError(f"Configuration for model {self.model_name} could not be fetched.")
        
        if not self.deployment_name:
            raise InvalidLLMModelError(f"Model {self.model_name} is not supported in Azure.")
        
        logger.info(f"Loaded Azure Model: {self.model_name}")

    def build_llm_object(self, llm_request, async_iter_callback_handler):
        """
        Builds and returns the Azure ChatOpenAI object.
        """
        try:
            response = AzureChatOpenAI(
                deployment_name=self.deployment_name,
                azure_endpoint=self.base_url,
                openai_api_key=self.api_key,
                openai_api_version=self.api_version,
                model=llm_request.llm_option,
                temperature=llm_request.temperature,
                streaming=True,
                callbacks=[async_iter_callback_handler]
            )
            logger.info(f"Model used: Azure {self.model_name}")
            return response
        except InvalidLLMModelError as error:
            return {"detail": str(error)}
        except GenericError as generic_error:
            return {"detail": str(generic_error)}

    def _call_llm(self,llm_request, async_iter_callback_handler):
        '''
        Calls the Azure Language Model based on the provided Request object.
        '''
        llm_object = self.build_llm_object(llm_request,async_iter_callback_handler)
        return llm_object
    

    async def get_llm_result(self, llm_request, search_request, op,response_type, chat_prompt, chain, cache_manager,chat_history, async_iter_callback_handler, retriever_results=None,  langfuse_handler=None, is_image=False, chain_span=None, trace=None):
        """
        Get the result by running the LLM chain with the provided request.
        """
        context_list = llm_request.context
        complete_response = ""
        task = None
        try:
            start_time = datetime.now()
            chain_input = {}
            if not is_image:
                chain_input = {
                    "query": [llm_request.query],
                    "chat_history": chat_history
                }
                if context_list is not None:
                    chain_input["content"] = context_list

            if langfuse_handler is None:
                task = asyncio.create_task(chain.ainvoke(chain_input))
            else:
                task = asyncio.create_task(chain.ainvoke(chain_input, config={"callbacks": [langfuse_handler]}))

            if task:
                yield f"event: response_type\ndata: {response_type}\n\n"
            
            async for token in async_iter_callback_handler.aiter():
                complete_response += token
                token = token.replace('\n', '\\n')
                yield f"event: llm_response\ndata: {token}\n\n"
            end_time = datetime.now()

        except Exception as err:
            logger.error(f"Error occurred during streaming: {err}")
            yield f"error: An Error has occurred while streaming the llm response.\n\n"
            raise
        finally:
            async_iter_callback_handler.done.set()

        if retriever_results is not None and search_request.is_kb_retrieval:
            if op =='maskedKB':
                masked_retriever_results=restructure_retriever_results(retriever_results)
                for data in masked_retriever_results:
                    yield f"event: response_source\ndata: {json.dumps(data)}\n\n"
            else:
                serialized_metadata = jsonpickle.encode(retriever_results)
                deserialized_metadata = json.loads(serialized_metadata)
                clean_metadata = [obj["py/state"]["__dict__"] for obj in deserialized_metadata]
                for data in clean_metadata:
                    temp_metadata = data['metadata']["py/state"]["__dict__"]
                    data['metadata'] = temp_metadata
                    yield f"event: response_source\ndata: {json.dumps(data)}\n\n"

        trace_id = ""
        trace_url = ""

        if trace is not None:
            trace_id = trace.id
            trace_url = trace.get_trace_url()
            trace_url = update_trace_url(trace_url)
            trace.update(output=complete_response)
       
        yield f"event: trace_id\ndata: {trace_id}\n\n"
        yield f"event: trace_url\ndata: {trace_url}\n\n"
        yield f"event: conversation_id\ndata: {str(cache_manager.key_id)}\n\n"
        
        langfuse_input_output_cost_update(trace_id, llm_request, search_request, op, chat_history, complete_response, is_image, chain_span, chat_prompt, trace, start_time, end_time)

        cache_manager.add_conversation(human_msg=llm_request.query, ai_msg=complete_response)
        await task

    @timer_func
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    #get_llm_result_with_streaming
    async def get_llm_result_with_streaming(self, llm_request,
                             search_request,
                             op: str,
                             response_type,
                             chain_1_info: dict, 
                             cache_manager,
                             chat_history=None, 
                             retriever_results=None,
                             trace=None,
                             is_image=False):
    
        """
        Call the LLM model with the specified request.
        """
        global RETRY_COUNT
        RETRY_COUNT += 1
        if trace is not None:
            trace.update(metadata={"LLM Chain Max retries": RETRY_COUNT})

        try:
            logger.debug("Calling LLM with request: %s", llm_request)

            async_iter_callback_handler = AsyncIteratorCallbackHandler()
            
            llm = self._call_llm(llm_request, async_iter_callback_handler)

            langfuse_handler = None
            chain_span=None
            if trace is not None:
                langfuse_handler = trace.get_langchain_handler()
                
            if is_image is True:
                chat_prompt = create_chat_prompt_for_images(llm_request,search_request,op,chat_history)
                chain_span = trace.span(
                    name="RunnableSequence",
                    start_time=datetime.now(),
                    input=llm_request.query
                )
                langfuse_handler=None
                
            else:
                
                chat_prompt = create_chat_prompt(llm_request, op)

            chain = chat_prompt | llm

            return self.get_llm_result(llm_request, search_request, op, response_type,chat_prompt, chain, cache_manager,chat_history, async_iter_callback_handler, retriever_results,  langfuse_handler, is_image,chain_span, trace)

        except Exception as error:
            logger.error("Error in get_llm_result_with_streaming: %s\n%s", str(error), traceback.format_exc())
            logger.info("Azure Open AI Check Retry -------------->>>>>> %s", RETRY_COUNT)
            raise


