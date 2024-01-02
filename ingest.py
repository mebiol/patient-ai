from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = 'data/'
DB_FIASS_PATH = 'vectorstores/db_faiss'

#create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.csv', loader_cls=CSVLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', 
                                          model_kwargs = {'device':'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FIASS_PATH)
    
if __name__ == '__main__':
    create_vector_db()
