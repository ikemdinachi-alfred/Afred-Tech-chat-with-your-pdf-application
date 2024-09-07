import os

import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_spliter.split_text(text)
    return chunks


# Function to create FAISS index from text chunks and store in session state
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Store FAISS index in session state for reuse during the session
    st.session_state['faiss_index'] = faiss_index


# Function to retrieve the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    say 'Answer rephrase to help me find answer for you in the right context.' Please do not make up information. \n\n
    Context: \n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to process user input and respond based on FAISS search
def user_input(user_question):
    if "faiss_index" not in st.session_state:
        st.error("Please upload and process PDF files first.")
        return

    # Retrieve the FAISS index from session state
    faiss_index = st.session_state['faiss_index']
    docs = faiss_index.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def safe_run(func):
    try:
        result = func()
        return result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Log the error for further investigation if needed
        print(f"Error: {str(e)}")


# Main function to run the Streamlit app
def main():
    # Set the page configuration
    st.set_page_config(page_title="Alfred-Tech PDF Chat", layout="wide")

    # Add custom styles for the header and footer
    st.markdown(
        """
        <style>
        /* Background gradient for the page */
        body {
            background: linear-gradient(to bottom right, #f0f4c3, #b3e5fc);
        }

        /* Alfred-Tech header styling */
        h1 {
            text-align: center;
            font-size: 52px;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: #f57c00;
            text-shadow: 2px 2px 5px #ffa726;
        }

        /* PDF Chat header styling */
        h2 {
            text-align: center;
            font-size: 36px;
            font-family: 'Arial', sans-serif;
            color: #1976d2;
            text-shadow: 2px 2px 5px #64b5f6;
        }

        /* Styling for the book icon */
        img {
            vertical-align: middle;
            margin-left: 10px;
        }

        /* Center content in the page */
        .stApp {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        /* Footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1976d2;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }

        /* Decorative heart icon */
        .heart {
            color: red;
            font-size: 22px;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add the headers with custom styles and a book icon
    st.markdown(
        """
        <h1>Alfred-Tech</h1>
        <h2>PDF Chat <img src="https://cdn-icons-png.flaticon.com/512/3081/3081392.png" width="40" height="40"></h2>
        """,
        unsafe_allow_html=True
    )

    # Add a footer with dedication
    st.markdown(
        """
        <div class="footer">
            <p>Dedicated to Mrs. F. Oluwatosin with love <span class="heart">❤️</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDF files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF files processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

    # User input field to ask questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    result = safe_run(lambda: user_input(user_question))
    if result:
        st.write(result)


if __name__ == "__main__":
    main()
