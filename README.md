Here's a comprehensive, yet simple, README file for your Streamlit PDF chat application:

---

# Alfred-Tech PDF Chat Application

Welcome to **Alfred-Tech PDF Chat**, an application that allows users to upload PDF documents and ask questions about their content in real time. This app uses **Google Generative AI** for question-answering and **FAISS(Facebook Ai Similarity Search)** for efficient document search. The application is built using **Streamlit** for easy deployment.

---

## Features

- Upload and process multiple PDF files.
- Efficiently search through PDF content using FAISS indexing.
- Ask questions about the PDFs and get detailed responses.
- Real-time chat interface for document interaction.

---

## Prerequisites

Make sure you have the following installed:

1. **Python 3.8+**
2. **pip** (Python package manager)
3. **Google Generative AI API Key**
4. **Streamlit Cloud or Local Deployment Setup**

---

## Setup and Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/alfred-tech-pdf-chat.git
   cd alfred-tech-pdf-chat
   ```

2. **Set up a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   Create a `.env` file in the project root and add your **Google Generative AI API Key**:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Run the application locally**:

   ```bash
   streamlit run app.py
   ```

   This will launch the app in your default web browser.

---

## How to Use

1. **Upload PDF Files**:
   - On the sidebar, upload one or more PDF files by clicking the "Upload your PDF Files" button.
   - Once uploaded, click **Process** to extract and index the content from your PDFs.

2. **Ask a Question**:
   - After processing the PDFs, use the input field at the center of the page to type your question about the PDF content.
   - The app will search the content and return the most relevant answer.

---

## Key Components

- **Streamlit**: Provides the user interface and interaction.
- **FAISS**: A fast document search index to retrieve the most relevant chunks of the PDF text.
- **Google Generative AI**: Provides answers to the questions based on the retrieved document context.

---



## Limitations

- This app runs on a non-persistent server (Streamlit Cloud) which means that the PDF data is only available during the session.
- The app doesn't store PDF data or user queries across sessions.
- The FAISS index is generated in-memory and is lost when the session ends.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests to improve the functionality.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For any inquiries or suggestions, email us  "ikemdinachialfred@gmail.com".

---

### Enjoy using **Alfred-Tech PDF Chat** to get real-time insights from your PDF files!
### Using **https://pdfchart.streamlit.app**
---
