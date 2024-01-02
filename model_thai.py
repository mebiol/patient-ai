from langchain.prompts import PromptTemplate
from langchain .embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
import requests
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
# import chainlit  as cl

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    # tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt-1.0.0-beta-13b-chat-gguf")
    model = pipeline("text-generation", model="openthaigpt/openthaigpt-1.0.0-alpha-7b-chat-ckpt-hf")
    llm = CTransformers(
        model = model,
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response['result']  # This line is changed to return only the answer

def is_english(text):
    return not bool(re.search('[\u0E00-\u0E7F]', text))
                    
def translate(data):
    req = requests.get(f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=th&dt=t&q={requests.utils.quote(data)}")
    
    translated_data = req.json()[0][0][0]
    print(f"Translated from {translated_data} to English")
    return translated_data    

# Main execution function
def main():
    query = input("Enter your query: ")
    result = final_result(query)
    if not is_english(result):
        translate(result)
    else:    
        print("Response:", result)

if __name__ == "__main__":
    main()
