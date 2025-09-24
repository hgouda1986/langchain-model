# This script uses the GoogleGenerativeAIEmbeddings to get a text embedding.

# Before running, make sure you have the necessary libraries installed:
# pip install langchain-google-genai python-dotenv

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from a .env file
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))
# IMPORTANT: Ensure your Google API key is set as an environment variable.
# Example: export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: The GOOGLE_API_KEY environment variable is not set.")
    exit()

# Initialize the embedding model.
# The 'models/embedding-001' is a free-tier model.
# The `dimensions` parameter is not used for this model as it has a fixed output size.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Get the embedding for the query string.
result = embeddings.embed_query("Delhi is the capital of India")

# Print the resulting vector.
print(result)

# Print the length of the vector for confirmation.
print(f"\nVector dimensions: {len(result)}")