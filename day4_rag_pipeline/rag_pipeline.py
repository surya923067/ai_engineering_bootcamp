import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. LOCAL KNOWLEDGE BASE (Document Store) ---
documents = [
    "App crash ho raha hai jab main login button dabata hoon. Error code: 500.",
    "Payment failed ho gaya mera, paisa kat gaya lekin order confirm nahi hua. Refunds take 3-5 working days.",
    "Yeh app bohot hi shandaar hai, UI ekdum smooth hai aur loading time fast hai!",
    "Password reset karne ka option settings menu ke inside 'Security' tab mein milta hai.",
]

# --- 2. USER QUERY ---
user_query = "Mera payment kat gaya hai par order nahi mila, mera refund kab aayega?"
print(f"💬 User Query: '{user_query}'\n")

# --- 3. RETRIEVAL STEP (TF-IDF + Cosine Similarity) ---
print("🔍 Step 1: Performing Similarity Search on Knowledge Base...")
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([user_query])

# Calculate mathematical closeness
similarities = cosine_similarity(query_vector, doc_vectors).flatten()
top_match_idx = np.argmax(similarities)
retrieved_context = documents[top_match_idx]

print(f"✅ Top Retrieved Context (Index {top_match_idx}):")
print(f"   \"{retrieved_context}\"\n")

# --- 4. RAG GENERATION LOGIC ---
print("🚀 Step 2: RAG Pipeline successfully executed retrieval!")
print("🎯 Retrieved Context Ready for LLM Generation:")
print(f"--> Answer Formulation based on context: '{retrieved_context}'")
print("\n✨ [Day 4 Core Architecture Verified]: Retrieval-Augmented Generation (RAG) vector matching completed successfully without errors!")