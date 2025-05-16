import os
import traceback
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from google.genai import types
import pickle

# Hard-code your Google API key here
GEMINI_API_KEY = "Enter your key here"  # Replace with your actual API key
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# Load the FAISS index and embeddings
INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"

print("Loading embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)

print("Loading FAISS index...")
# Load the vector store
vectorstore = FAISS.load_local(
    INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """
You are a helpful legal assistant that answers questions based on the provided legal documents.
- If the user asks a general greeting or farewell, respond naturally and briefly.
- If the user asks a question about legal content, answer based ONLY on the provided document excerpts.
- If you don't find relevant information in the provided documents, say "I don't have information about that in my knowledge base."
- Keep your answers concise and focused on the legal information in the documents.
- Do not make up information or cite sources that weren't provided.
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        query_text = data.get("query", "").strip()
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        print(f"Processing query: {query_text}")
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query_text)
        
        # Extract content from documents
        context = ""
        sources = []
        
        for doc in docs:
            context += f"\n\nDocument: {doc.metadata.get('source', 'Unknown')}\n"
            context += doc.page_content
            
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown")
            })
        
        # If no documents were retrieved, still proceed (for conversational queries)
        if not context:
            context = "No relevant documents found."
        
        # Create the prompt with context
        prompt_text = f"""
{SYSTEM_PROMPT}

RELEVANT DOCUMENT EXCERPTS:
{context}

USER QUERY: {query_text}

Please provide a response based on the above information.
"""
        
        # Generate response using Gemini 2.0 Flash
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.1,
        )
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )
        
        answer = response.text.strip()
        
        # Determine if this is a conversational response
        is_conversation = len(sources) == 0 or len(query_text.split()) < 3
        
        # Check if the answer indicates no information was found
        no_info_phrases = [
            "i don't have", "i do not have", "no information", 
            "cannot find", "couldn't find", "could not find",
            "no relevant", "not mentioned", "not specified", "not provided"
        ]
        
        has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
        
        # If no relevant information was found and there are no sources, provide a clearer message
        if has_no_info and not sources:
            answer = "I don't have information about that in my knowledge base. I can only answer questions related to the legal documents I've been trained on."
            is_conversation = True
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "is_conversation": is_conversation
        })
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Add a general error handler
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)


