# Day 4: First-Principles RAG (Retrieval-Augmented Generation) Pipeline

## 🚀 Overview
In Day 4, we built a complete **RAG Pipeline** from first principles without relying on heavy or error-prone vector database installations (like ChromaDB on Windows). This architecture combines mathematical similarity search with a generative AI model to deliver accurate, context-aware answers.

## 🛠️ Tech Stack & Architecture
- **Language:** Python 3.15
- **Retrieval Engine:** Scikit-Learn (`TfidfVectorizer` + `Cosine Similarity`) for semantic/keyword matching.
- **Data Store:** Local Document List (Simulating Knowledge Base).
- **Generation Engine:** Gemini API (`gemini-2.5-flash`) via REST (`requests`).

## 📂 Code Workflow
1. **Knowledge Base:** Stores raw support tickets / customer feedback texts.
2. **Retrieval Step:** Converts the user query and documents into numerical vectors using TF-IDF and calculates cosine similarity to fetch the most relevant context chunk.
3. **Augmentation & Generation:** Injects the retrieved context into a strict prompt template and queries the Gemini API to formulate a hallucination-free answer.

## ▶️ How to Run
1. Ensure dependencies are installed:
   ```bash
   pip install numpy scikit-learn requests pandas