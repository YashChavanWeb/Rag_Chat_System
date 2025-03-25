import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pinecone
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pinecone initialization
from pinecone import Pinecone, ServerlessSpec

# Create a Pinecone instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create an index (if not exists)
if "test-index" not in pc.list_indexes().names():
    pc.create_index(
        name="test-index",
        dimension=768,  # Update this according to your embedding dimension
        metric="cosine",  # You can also use "euclidean" or "dotproduct"
        spec=ServerlessSpec(
            cloud="aws", region="us-east-1"
        ),  # Cloud and region can be updated as per your setup
    )
else:
    # Check current dimension and delete and recreate if needed.
    index_description = pc.describe_index("test-index")
    if index_description.dimension != 768:
        pc.delete_index("test-index")
        pc.create_index(
            name="test-index",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

# Access the Pinecone index
index = pc.Index("test-index")  # Specify your index name here


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create and save a vector store using Pinecone
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Prepare the vectors for Pinecone upsert
    vectors = []
    for i, chunk in enumerate(text_chunks):
        embedding = embeddings.embed_documents([chunk])[0]  # Corrected line
        vectors.append(
            {"id": f"vec_{i}", "values": embedding, "metadata": {"text": chunk}}
        )

    # Upsert vectors into Pinecone
    index.upsert(vectors=vectors, namespace="pdf_namespace")
    st.success("Vectors uploaded to Pinecone successfully.")


# Function to define the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to handle user input, search for similar documents, and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a query vector from the user's question
    query_vector = embeddings.embed_query(user_question)  # Corrected line

    # Query Pinecone to find the most similar documents
    response = index.query(
        namespace="pdf_namespace",
        vector=query_vector,
        top_k=3,
        include_values=True,
        include_metadata=True,
    )

    # Extract the top documents from the response
    docs = [
        Document(page_content=item["metadata"]["text"]) for item in response["matches"]
    ]  # Corrected line

    # Get the conversational chain for answering the question
    chain = get_conversational_chain()

    # Generate a response using the question-answering chain
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Display the response in Streamlit
    st.write("Reply: ", response["output_text"])


# Main function to set up the Streamlit app and handle the user interface
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)


# Run the Streamlit app
if __name__ == "__main__":
    main()
