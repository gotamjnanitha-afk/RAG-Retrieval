
import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Retrieval Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()


# ==========================================
# 2. APPLICATION TITLE
# ==========================================

st.title("📚 RAG Retrieval Assistant")

st.write(
    "Upload a PDF document and ask questions. "
    "The application retrieves relevant information "
    "and generates answers using AI."
)

st.divider()


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# 4. EXTRACT PDF TEXT
# ==========================================

def extract_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number + 1,
                        "source": uploaded_file.name
                    }
                )
            )

    return documents


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ==========================================
# 6. CREATE VECTOR DATABASE
# ==========================================

def create_vector_database(chunks, embeddings):

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# ==========================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_documents(vector_store, question):

    retrieved_docs = vector_store.similarity_search(
        question,
        k=3
    )

    return retrieved_docs


# ==========================================
# 8. GENERATE ANSWER USING GROQ
# ==========================================

def generate_answer(question, retrieved_docs):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "Groq API key is missing. "
            "Please add GROQ_API_KEY to your .env file."
        )

    if not retrieved_docs:

        return "No relevant information was found in the document."

    # Combine retrieved document chunks

    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page', 'Unknown')}: "
            f"{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Connect to Groq LLM

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=api_key
    )

    # Create prompt template

    prompt = ChatPromptTemplate.from_template(
        """
        You are a document question-answering assistant.

        Answer the question using ONLY the provided context.

        Do not add information from outside the document.

        If the answer is not present in the context,
        say:
        "The answer was not found in the uploaded document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Create RAG chain

    chain = prompt | llm

    # Generate response

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content


# ==========================================
# 9. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write("**Project:** RAG Retrieval System")
    st.write("**Language:** Python")
    st.write("**Framework:** Streamlit")
    st.write("**Vector Database:** FAISS")
    st.write("**Embedding Model:** MiniLM")
    st.write("**LLM Provider:** Groq")
    st.write("**LLM Model:** GPT-OSS 120B")

    st.divider()

    st.info(
        "Upload a PDF document to begin "
        "retrieving information."
    )


# ==========================================
# 10. DOCUMENT UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:

    if st.button("Process Document", type="primary"):

        try:

            with st.spinner("Reading PDF document..."):

                documents = extract_pdf(uploaded_file)

            if not documents:

                st.error(
                    "No readable text found in this PDF. "
                    "Please upload a text-based PDF."
                )

                st.stop()

            with st.spinner("Splitting document into chunks..."):

                chunks = split_documents(documents)

            if not chunks:

                st.error("No text chunks were created.")

                st.stop()

            with st.spinner("Loading embedding model..."):

                embeddings = load_embedding_model()

            with st.spinner("Creating FAISS vector database..."):

                vector_store = create_vector_database(
                    chunks,
                    embeddings
                )

            # Store results in Streamlit session

            st.session_state.vector_store = vector_store

            st.session_state.document_name = uploaded_file.name

            st.session_state.chunk_count = len(chunks)

            st.session_state.page_count = len(documents)

            st.success("Document processed successfully!")

            # Display document statistics

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "PDF Pages",
                len(documents)
            )

            col2.metric(
                "Text Chunks",
                len(chunks)
            )

            col3.metric(
                "Vector Database",
                "Ready"
            )

        except Exception as e:

            st.error(f"Processing error: {e}")


# ==========================================
# 11. QUESTION AND ANSWER SECTION
# ==========================================

st.divider()

st.subheader("💬 Ask Questions About Your Document")


if "vector_store" in st.session_state:

    st.success(
        f"Current document: {st.session_state.document_name}"
    )

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is Artificial Intelligence?"
    )

    if st.button("Get Answer", type="primary"):

        if question.strip():

            try:

                # Retrieve relevant chunks

                with st.spinner("Retrieving relevant information..."):

                    retrieved_docs = retrieve_documents(
                        st.session_state.vector_store,
                        question
                    )

                # Generate answer

                with st.spinner("Generating AI answer..."):

                    answer = generate_answer(
                        question,
                        retrieved_docs
                    )

                # Display answer

                st.subheader("🤖 AI Answer")

                st.write(answer)

                # Display retrieved information

                st.subheader("📖 Retrieved Information")

                for index, doc in enumerate(retrieved_docs, 1):

                    with st.expander(
                        f"Retrieved Chunk {index} "
                        f"(Page {doc.metadata.get('page', 'Unknown')})"
                    ):

                        st.write(doc.page_content)

            except Exception as e:

                st.error(
                    f"Answer generation error: {e}"
                )

        else:

            st.warning("Please enter a question.")

else:

    st.info(
        "Please upload and process a PDF document "
        "before asking questions."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Understanding RAG Retrieval | "
    "Python + Streamlit + LangChain + FAISS"
)
