import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from helpers import format_docs

# Configure environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Set up persistent directories
UPLOAD_FOLDER = os.path.join('static', 'uploads')
CHROMA_PERSIST_DIR = os.path.join('static', 'chroma_db')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Anda adalah asisten yang berpengetahuan luas. Berdasarkan konteks berikut, jawablah pertanyaan dengan seakurat mungkin. 
    Buatkan baris baru \n setiap ada point-point

    Konteks: {context}
    Pertanyaan: {question}
    
    Jawaban:
    """,
)

llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize global variables
vectorstore = None
retriever = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global vectorstore
    global retriever
    
    if request.method == "POST":
        file = request.files['pdfFile']
        if file and file.filename.endswith('.pdf'):
            try:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                
                # Load the PDF
                loader = PyMuPDFLoader(file_path=file_path)
                data = loader.load()
                
                # Initialize Chroma with persistence
                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma.from_documents(
                    documents=data,
                    embedding=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR
                )
                
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}
                )
                
                return render_template("chat.html")
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Only PDF files are allowed"}), 400

@app.route("/rag", methods=["GET"])
def rag():
    global vectorstore
    global retriever
    
    if vectorstore is None or retriever is None:
        return jsonify({
            "status": {
                "code": 500,
                "message": "Retriever not initialized. Please upload a PDF first.",
            },
            "data": None,
        }), 500

    msg = request.args.get('msg')
    if not msg:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Message parameter 'msg' is required.",
            },
            "data": None,
        }), 400

    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    def generate():
        try:
            for chunk in rag_chain.stream(msg):
                if isinstance(chunk, str):
                    yield f"data: {chunk}\n\n".encode('utf-8')
                else:
                    yield b"data: " + chunk + b"\n\n"
            yield b"data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n".encode('utf-8')

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream'
    )

if __name__ == "__main__":
    app.run(debug=True)