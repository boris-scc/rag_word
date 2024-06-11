import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import (AzureAIDocumentIntelligenceLoader,
                                                  DirectoryLoader, UnstructuredWordDocumentLoader)
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import AzureSearch
from operator import itemgetter


load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
# os.environ["AZURE_OPENAI_ENDPOINT_USA"] = os.getenv("AZURE_OPENAI_ENDPOINT_USA")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")


file_path = r'C:\Users\BDAPNET\Downloads\test_data\Profil_SCC_CARTIER_Louis_2024_S1.docx'
# Initiate Azure AI Document Intelligence to load the document.
# You can either specify file_path or url_path to load the document.
# path = r'C:\Users\BDAPNET\Downloads\2024'
# folder_path = DirectoryLoader(path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
# load = folder_path.load()
loader = AzureAIDocumentIntelligenceLoader(
    file_path=file_path,
    api_key=os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_KEY'),
    api_endpoint=doc_intelligence_endpoint,
    api_model="prebuilt-layout",
    api_version="2023-10-31-preview"
)
docs = loader.load()

# Split the document into chunks base on markdown headers.
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]
# text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
#
# docs_string = docs[0].page_content
# splits = text_splitter.split_text(docs_string)


#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print("Length of splits: " + str(len(splits)))


# Embed the splitted documents and insert into Azure Search vector store

aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="textEmbed",
    openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
)

vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

index_name: str = "search-doc-word"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=aoai_embeddings.embed_query
)

vector_store.add_documents(documents=splits)


# Retrieve relevant chunks based on the question

retriever = vector_store.as_retriever(search_type="similarity")

retrieved_docs = retriever.invoke(
    "quelle est le niveau d'anglais"
)

# Use a prompt for RAG that is checked into the LangChain prompt hub
# (https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=989ad331-949f-4bac-9694-660074a208a7)
prompt = hub.pull("rlm/rag-prompt")
llm = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
    azure_deployment="gpt-4-32k",
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = rag_chain.invoke("quelle est son niveau anglais?")
print(res)


rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

result = rag_chain_with_source.invoke("quelle est sa formation?")
print(result)