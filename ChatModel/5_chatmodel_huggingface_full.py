# This script demonstrates how to use LangChain's chat model integration
# for a conversational model like TinyLlama.

import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from the .env file.
# This ensures your API token is correctly loaded.
load_dotenv()

# --- 1. Initialize the HuggingFaceEndpoint ---
# We use the HuggingFaceEndpoint to connect to the model on the Hub.
# Note: While `task="text-generation"` works, a more appropriate task
# for a chat model would be "conversational" if available, or simply
# using ChatHuggingFace to handle the chat logic.
# The `HuggingFaceEndpoint` class implicitly handles the API call details.
# We also explicitly pass the API token here to be safe.
try:
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
except Exception as e:
    print(f"Error initializing HuggingFaceEndpoint: {e}")
    print("Please ensure your HUGGINGFACEHUB_API_TOKEN is set correctly.")
    exit()

# --- 2. Wrap the LLM in a ChatHuggingFace object ---
# This is the standard way to prepare an LLM for chat-based interactions in LangChain.
chat_model = ChatHuggingFace(llm=llm)

# --- 3. Define the Chat Prompt Template ---
# Instead of a simple string, we use a structured template with roles.
# This is the best practice for conversational models.
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant. You answer all questions concisely and politely."
        ),
        HumanMessage(content="What is the capital of {country}?"),
    ]
)

# --- 4. Chain the Prompt and Model ---
# This creates a simple pipeline to pass the prompt to the model.
chain = prompt | chat_model

# --- 5. Invoke the chain and get the result ---
# We invoke the chain with the specific value for the 'country' variable.
print("Generating response...")
result = chain.invoke({"country": "India"})

# The result is an AIMessage object. We can access its content.
print("\nGenerated response:")
print(result.content)
