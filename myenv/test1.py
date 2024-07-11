import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
#from langchain_openai import ChatOpenAI
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import CSVLoader

mistral_api_key = 'VO1WqLGTxQOFZi5jpjkwj5j0Ca0O04BI'
nomic_apikey = 'nk-Ei7TKcGktODsxoXYo4PdXklOO5ZlVfmoQEp77W-UHO4'

def query_pdf(query, log):
    loader = CSVLoader("BalanceSheetSara.csv")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = NomicEmbeddings(model='nomic-embed-text-v1', nomic_api_key=nomic_apikey)
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index_machine")
    persisted_vectorstore = FAISS.load_local("faiss_index_machine", embeddings, allow_dangerous_deserialization=True)


    #from langchain.llms import HuggingFaceHub
    # HUGGINGFACEHUB_API_TOKEN='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'
    # models_llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
    #                             model_kwargs={"temperature":0.5,"max_new_tokens":512,"max_length":64},
    #                             huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    #pip install llama-cpp-python                       
    # from langchain.llms import LlamaCpp

    # path1="C:/Users/sony/Downloads/zephyr-7b-beta.Q4_0.gguf"
    # models_llm = LlamaCpp(
    #         streaming = True,
    #         model_path=path1,
    #         n_gpu_layers=2,
    #         n_batch=512,
    #         temperature=0.75,
    #         top_p=1,
    #         verbose=True,
    #         n_ctx=4096
    #         )

    # HUGGINGFACEHUB_API_TOKEN='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'
    # models_llm = HuggingFaceHub(repo_id="google/flan-t5-xl",
    #                             model_kwargs={"temperature":0.5,"max_new_tokens":512,"max_length":64},
    # huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)    



#     # pip install langchain-anthropic                         
#     from langchain_anthropic import ChatAnthropic

#     ANTHROPIC_API_KEY='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'
#     models_llm = ChatAnthropic(
#     model="Claude 2",
#     api_key=ANTHROPIC_API_KEY,
#     temperature=0
# )
    #pip install openllm
    # from langchain_community.llms import OpenLLM

    # models_llm = OpenLLM(
    #     model_name="dolly-v2",
    #     server_url="http://127.0.0.1:7860/",
    #     model_id="databricks/dolly-v2-3b",
    #     temperature=0.94,
    #     repetition_penalty=1.2,
    # )

    #pip install langchain-together
    # from langchain_together import ChatTogether
    # together_api_key='5ef993ceeabda364e86f855bca80b4ea92ef3aaea3de3422811461e24ccf3893'
    # models_llm = ChatTogether(
    #     together_api_key=together_api_key,
    #     model="meta-llama/Llama-3-70b-chat-hf",
    # )

    #from langchain_together import ChatTogether
    # together_api_key='5ef993ceeabda364e86f855bca80b4ea92ef3aaea3de3422811461e24ccf3893'
    # models_llm = ChatTogether(
    #     together_api_key=together_api_key,
    #     model="microsoft/phi-2",  ####coding model
    # )
    
    # together_api_key='5ef993ceeabda364e86f855bca80b4ea92ef3aaea3de3422811461e24ccf3893'
    # models_llm = ChatTogether(
    #     together_api_key=together_api_key,
    #     model="codellama/CodeLlama-70b-Python-hf") ####coding model

    #pip install ollama
    # from langchain_community.llms import Ollama
    # models_llm= Ollama(model="llama2")

    #pip install langchain_google_genai
    from langchain_google_genai import ChatGoogleGenerativeAI

    models_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key='AIzaSyB8xpvBG4IghqENvxxj2mV3vfnGNiZjras' )


    prompt_template = """
    Answer the following query only based on the provided context.
    Answer the question based only on the following context:
    {context}
    When numeric values or dates are involved, ensure that the answer reflects the appropriate interpretation and calculation from the retrieved context.
    when context is missing then consider that value as zero.
    Do not provide any information that is not contained in the context. 
    If the answer is not found in the context, respond with: "The answer to the query is not mentioned in the context of the CSV document."

    Query: {query}
    """



#     prompt_template = """
#     Answer the query using the tabular data in the uploaded document. Consider all relevant rows and data types (numeric, string, date). Understand and apply keywords such as "top," "highest," "lowest," "below," etc., in basic English while responding. If the answer is not found in the document, respond with: "The answer to the query is not mentioned in the context of the document."

#     **Query:** {query}

#     **Instructions:**
#     1. Search the uploaded document for rows relevant to the query.
#     2. Extract and analyze the relevant data from these rows, considering all data types.
#     3. Summarize or aggregate the data as necessary to answer the query.
#     4. Handle terms like "top," "highest," "lowest" by providing the required sorted or filtered data.
#     5. Specify the criteria used for filtering, sorting, or aggregation in your response.

#     **Examples:**
#     - For "highest sales": Provide the row with the maximum sales value.
#     - For "top 3": List the top 3 entries sorted by the relevant field.
#     - For "below 100": Filter and list all entries with values less than 100.
# """



    def strict_prompt(query):
        return prompt_template.format(query=query)
    
#     retriever=persisted_vectorstore.as_retriever()

#     chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt_template
#     | models_llm
#     | StrOutputParser()
# )


    qa = RetrievalQA.from_chain_type(llm=models_llm, chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
    result = qa.run(query)

    if not result.strip() or "The answer to the query is not mentioned in the context of the document." in result:
        result = "The answer to the query is not mentioned in the context of the document."

    log.append((query, result))
    return result, log

def gradio_interface(query, log):
    response, log = query_pdf(query, log)
    return response, update_html(log), log

def update_html(log):
    rows = "".join([f"<tr><td style='border: 1px solid black; padding: 5px;'>{q}</td><td style='border: 1px solid black; padding: 5px;'>{r}</td></tr>" for q, r in log])
    return f"""
    <table style="width:100%; border-collapse: collapse; border: 1px solid black;">
        <thead>
            <tr>
                <th style='border: 1px solid black; padding: 5px;'>Query</th>
                <th style='border: 1px solid black; padding: 5px;'>Output</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="QUERY"), gr.State([])],  
    outputs=[gr.Textbox(label="OUTPUT"), gr.HTML(label="Log"), gr.State([])],
    title="CSV ChatBot"
)

if __name__ == "__main__":
    interface.launch()
