# Langchain Document Loaders: [https://python.langchain.com/docs/integrations/document_loaders/]
# Langchain Text Splitters: [https://python.langchain.com/docs/concepts/text_splitters/]
# Recursive Character Text Splitter doc: [https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html]
# Embedding Generation options: [https://python.langchain.com/docs/integrations/text_embedding/]
# Ollama Chat Model: [https://python.langchain.com/docs/integrations/llms/ollama/]

from langchain_community.document_loaders import PyPDFDirectoryLoader    # langchain.document_loader depricated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain_community.embeddings.bedrock import BedrockEmbeddings    # aws embeddings
from langchain_ollama import OllamaEmbeddings    # ollama embeddings
from langchain_chroma import Chroma
from langchain.prompts import  ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

# CONSTANTS
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_tutorial"
DATA_PATH = "./pdfs"


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):    # type hinting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)


# pay-as-you-go service
# def get_embedding():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default", region_name="us-east-1"
#     )

#     return embeddings


def get_embedding():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")    # best embedding model avialable in ollama
    return embeddings


# attaching different ids to each chunk, for DB update later on
def calculate_chunk_ids(chunks):
    curr_chunk_idx = 0
    prev_page_id = ""

    for chunk in chunks:
        curr_page_id = f"{chunk.metadata['source']}:{chunk.metadata['page']}"
        
        if curr_page_id == prev_page_id:
            curr_chunk_idx += 1
        else:
            curr_chunk_idx = 0    # reset

        prev_page_id = curr_page_id
        chunk_id = f"{curr_page_id}:{curr_chunk_idx}"

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )
    
    curr_items = db.get(include=[])    # ids included by default
    curr_ids = set(curr_items["ids"])
    print(f"[INFO] Current Documents: {len(curr_items['ids'])}.")

    # adding docs not in db
    new_chunks = []
    new_chunk_ids = []
    for chunk in chunks:
        if not chunk.page_content.strip():
            print(f"[WARNING] Empty content for chunk id: {chunk.metadata['id']}")

        else:
            if chunk.metadata['id'] not in curr_ids:
                new_chunks.append(chunk)
                new_chunk_ids.append(chunk.metadata['id'])

    if new_chunks:
        db.add_documents(new_chunks, ids=new_chunk_ids)

    else:
        print("[ERROR] NO VALID CHUNKS TO ADD!")
    # db.persist() [automatically done in newer versions]

    print(f"[INFO] Newly Added Documents: {len(new_chunks)}.")


def query_rag(query_txt: str, model_name: str):

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )

    PROMPT_TEMPLATE = """
    Answer any question only on the following context:
    {context}

    ---
    Answer the question based on the above context: {question}
    """

    # retrieve most relevant chunks to our question
    results = db.similarity_search_with_score(query_txt, k=5)     

    context_txt = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_txt, question=query_txt)
    # print(prompt)

    model = OllamaLLM(model=model_name)
    response_txt = model.invoke(prompt)
    print(f"[RESULT] {response_txt}")

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"[RESULT] {sources}")


def reset_collection():
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )
    # before_existing = db.get(include=[])
    # print(f"[DEBUG] Document IDs (Before Reset): {before_existing['ids']}")

    db.delete_collection()
    print(f"[INFO] Chroma Collection '{COLLECTION_NAME}' has been reset.")

    # db = Chroma(
    #     collection_name=COLLECTION_NAME,
    #     embedding_function=get_embedding(),
    #     persist_directory=CHROMA_PATH
    # )

    # after_existing = db.get(include=[])
    # print(f"[DEBUG] Document IDs (After Reset): {after_existing['ids']}")


def main():
    # defining arguments
    parser = argparse.ArgumentParser(description="Local RAG Pdf(s) QnA")
    parser.add_argument("--query", type=str, required=True, help="Query you want to ask")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama Model ID for Response Generation")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma Collection before indexing")
    # store_true: if argument is present, store true else false

    args = parser.parse_args()

    if args.reset:
        reset_collection()

    documents = load_documents()

    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    add_to_chroma(chunks_with_ids)

    # query_txt = """
    #     Are there any clauses which defines that I cannot open another work for myself
    #     while working at TCS Research.
    # """
    query_rag(query_txt=args.query, model_name=args.model)
    # pprint.pp(model_response)


# ensures that script executes only when run directly
if __name__ == "__main__":
    main()