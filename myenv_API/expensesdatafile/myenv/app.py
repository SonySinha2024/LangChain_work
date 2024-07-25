## Importing Libraries
import gradio as gr
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI

## Configuration
config_file_path ='C:\genaiCheck\expensesdatafile\myenv\.env'
load_dotenv(dotenv_path=config_file_path)
google_api_key=os.environ.get('google_api_key')
nomic_apikey=os.environ.get('nomic_apikey')

## Function to read CSV File
def query_pdf(query):
    loader = CSVLoader("ExpensesData.csv")
    documents = loader.load()
    
    ## Text Splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    ## Nomic Embeddings
    embeddings = NomicEmbeddings(model='nomic-embed-text-v1',nomic_api_key=nomic_apikey)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
   
    ## Vector DB (Local File)
    vectorstore.save_local("faiss_index_machine")
    persisted_vectorstore = FAISS.load_local("faiss_index_machine",embeddings, allow_dangerous_deserialization=True)

    ## Google Generative AI Model
    models_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=google_api_key)

    ## System Prompt
    prompt_template = """
    Consider youself as assistant who can perform complex mathematical functions like finding the totals, calculating the 
    average, finding the maximum and minimum values, finding the top ten highest expenses, credits, debits amount.
    Answer the following query only based on the provided context.
    Answer the question based only on the following context:
    {context}
    When numeric values or dates are involved, ensure that the answer reflects the appropriate interpretation and calculation from the retrieved context.
    when context is missing then consider that value as zero.
    Do not provide any information that is not contained in the context. 
    If the answer is not found in the context, respond with: "The answer to the query is not mentioned in the context of the CSV document."

    Query: {query}
    """
     
    ## Prompt with Query
    def strict_prompt(query):
        return prompt_template.format(query=query)
    
#     retriever=persisted_vectorstore.as_retriever()

#     chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt_template
#     | models_llm
#     | StrOutputParser()
# )

    ## RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=models_llm, chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
    result = qa.run(query)

    ## Handling empty or not found results
    if not result.strip() or "The answer to the query is not mentioned in the context of the document." in result:
        result = "The answer to the query is not mentioned in the context of the document."

    ## Appending Query and Result to the log
    #log.append((query, result))
    #return result, log
    return result


## Gradio Interface
def gradio_interface(query, log):
    response, log = query_pdf(query, log)
    return response, update_html(log), log

## Create HTML Log for each Query and Result entry log
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

## Creating Gradio Interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="QUERY"), gr.State([])],  
    outputs=[gr.Textbox(label="OUTPUT"), gr.HTML(label="Log"), gr.State([])],
    title="CSV ChatBot"
)

## Launching the Interface
if __name__ == "__main__":
    interface.launch()

