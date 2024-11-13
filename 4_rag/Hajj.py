import os
import fitz
import psycopg2
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStore

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document  # Import Document class
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# Load environment variables from .env
load_dotenv()

# Set up PostgreSQL connection
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),  # Set this to your database name, e.g., 'hajj_db'
    user=os.getenv("DB_USER"),    # Your PostgreSQL username
    password=os.getenv("DB_PASSWORD"),  # Your PostgreSQL password
    host=os.getenv("DB_HOST", "localhost")  # Default to localhost if not specified
)
cursor = conn.cursor()

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Define the directory containing the text file and the persistent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "books", "hajj.pdf")
# persistent_directory = os.path.join(current_dir, "db","chroma_hajj") [Used when using chromaDB vectore store]

# Function to read text from a PDF file
def load_pdf_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
    return text

# Function to process PDF and store embeddings in PostgreSQL
def process_and_store_pdf(file_path):
    # Load and split the PDF text into chunks
    documents = load_pdf_text(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = text_splitter.split_text(documents)
    docs = [Document(page_content=chunk) for chunk in doc_chunks]

    # Store each chunk's embedding in PostgreSQL
    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        cursor.execute(
            "INSERT INTO hajj_embeddings (content, embedding) VALUES (%s, %s)",
            (doc.page_content, vector)
        )
    conn.commit()

# Function to retrieve similar documents using PostgreSQL and pgvector
def retrieve_similar_docs(query):
    # Embed the query
    query_vector = embeddings.embed_query(query)
    
    # Query PostgreSQL to find the most similar documents
    cursor.execute(
        """
        SELECT content
        FROM hajj_embeddings
        ORDER BY embedding <-> %s
        LIMIT 3
        """,
        (query_vector,)
    )
    results = cursor.fetchall()
    return [result[0] for result in results]

# Define the main process for embedding storage and retrieval
def main():
    # Path to the PDF file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books", "hajj.pdf")

    # Process and store PDF contents in the database
    process_and_store_pdf(file_path)

    # Example similarity query
    query = "What is hajj?"
    relevant_docs = retrieve_similar_docs(query)
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc}\n")

# Custom Retriever Class
class CustomRetriever:
    def __init__(self, retrieval_function):
        self.retrieval_function = retrieval_function

    def get_relevant_documents(self, query):
        # Call the retrieval function to get relevant documents
        return self.retrieval_function(query)

# Instantiate the custom retriever with `retrieve_similar_docs` as the retrieval function
custom_retriever = CustomRetriever(retrieve_similar_docs)

#______________________[Used when using chromaDB vectore store]______________________________________
# # Check if the Chroma vector store already exists
# if not os.path.exists(persistent_directory):
#     print("Persistent directory does not exist. Initializing vector store...")

#     # Ensure the text file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"The file {file_path} does not exist. Please check the path."
#         )
#     # Function to read text from a PDF file
#     def load_pdf_text(file_path):
#         text = ""
#         with fitz.open(file_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") # type: ignore # Extract text from each page
#         return text

#     # Load the PDF text data
#     documents = load_pdf_text(file_path)

#     # Split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     doc_chunks = text_splitter.split_text(documents)
#     # Convert each chunk into a Document object
#     docs = [Document(page_content=chunk) for chunk in doc_chunks]


#     # Display information about the split documents
#     print("\n--- Document Chunks Information ---")
#     print(f"Number of document chunks: {len(docs)}")
#     print(f"Sample chunk:\n{docs[0]}\n")

#     # Create embeddings
#     print("\n--- Creating embeddings ---")
#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
#     )  # Update to a valid embedding model if needed
#     print("\n--- Finished creating embeddings ---")

#     # Create the vector store and persist it automatically
#     print("\n--- Creating vector store ---")
#     db = Chroma.from_documents(
#         docs, embeddings, persist_directory=persistent_directory)
#     print("\n--- Finished creating vector store ---")

# else:
#     print("Vector store already exists. No need to initialize.")          
# ____________________________________________________________________________________






#______________________[Used when using chromaDB vectore store]______________________________________
# Load the existing vector store with the embedding function
# db = Chroma(persist_directory=persistent_directory,
#             embedding_function=embeddings)
# _____________________________________________________________________________________________________


# Define the user's question
query = "What is hajj?"

#______________________[Used when using chromaDB vectore store]______________________________________
# # Create a retriever for querying the vector store
# # `search_type` specifies the type of search (e.g., similarity)
# # `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )
# _____________________________________________________________________________________________________


#_______________Checking if the model is retrieving relevant doc or not_______________________
# relevant_docs = retriever.invoke(query)

# # Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
# _______________________________________________________________________________________________________

# Create a ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Pass the retrieval function directly
# history_aware_retriever = create_history_aware_retriever(
#     llm=llm,
#     retriever=retrieve_similar_docs,  # Pass function directly
#     context_prompt=contextualize_q_prompt
# )

#______________________[Used when using chromaDB vectore store]______________________________________
# # Create a history-aware retriever
# # This uses the LLM to help reformulate the question based on chat history
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )
# _______________________________________________________________________________________________________

# Function to contextualize the question based on chat history
def contextualize_question(query, chat_history):
    # Prepare the input for contextualization
    prompt_input = {
        "input": query,
        "chat_history": chat_history
    }
    # Format the prompt with the input values
    formatted_prompt = contextualize_q_prompt.format(**prompt_input)
    # Pass the formatted prompt to the language model to get the reformulated question
    reformulated_question = llm.predict(formatted_prompt)
    return reformulated_question

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise. Always answer in Bengali language"
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(custom_retriever, question_answer_chain)


# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
       
        # Contextualize the question before retrieving documents
        reformulated_query = contextualize_question(query, chat_history)

        # Pass the reformulated query to the RAG chain
        result = rag_chain.invoke({
            "input": reformulated_query,
            "chat_history": chat_history
        })

        # Display the AI's response
        print(f"AI: {result['answer']}")

        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))


# Main function to start the continual chat and process PDF
if __name__ == "__main__":
    main()  # Process PDF and store embeddings
    continual_chat()  # Start interactive chat
    conn.close()  # Close the database connection