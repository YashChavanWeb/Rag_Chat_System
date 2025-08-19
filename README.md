### RAG Chat System

This project is a Retrieval-Augmented Generation (RAG) system designed to interact with document data. The system uses a large language model to answer questions by first retrieving relevant information from a collection of documents (PDFs) and then generating a comprehensive response based on the retrieved context.

**Key Features:**
* **Document-Based Q&A:** Answers questions using information extracted from PDF files.
* **Efficient Retrieval:** Utilizes an index (`faiss_index`) for fast and efficient retrieval of relevant text from the documents.
* **Scalable Architecture:** Capable of handling a large collection of documents.

**Project Structure:**
* `app.py`: The main application file.
* `Textual Data`: Directory containing the source PDF documents.
* `faiss_index`: Stores the index created from the documents for retrieval.
* `requirements.txt`: Lists all necessary dependencies.

