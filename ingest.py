from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,WebBaseLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'context.txt'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db(type_loader):
    if type_loader == 'web_page':
        web_links = ["https://www.buffalo.edu/studentlife/new-to-ub/first-year-and-transfer-students/faqs-orientation.html"]
        loader = WebBaseLoader(web_links)
        documents = loader.load()    
    if type_loader == 'text':
        loader = TextLoader(DATA_PATH)
        documents = loader.load()
    if type_loader == 'pdf':
        loader = DirectoryLoader(DATA_PATH,
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)

        documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

