Of course. Based on the detailed information you provided, here is a comprehensive and professionally formatted `README.md` file for your GitHub repository.

***

# NutriChat: A RAG-Powered Chatbot for PDF Documents

[
[
[

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows you to chat with your PDF documents. This project demonstrates how to use local LLMs and state-of-the-art embedding models to build a powerful, factual, and context-aware question-answering system.

## Table of Contents
*   [Overview](#overview)
*   [Why Use RAG?](#why-use-rag)
*   [How It Works](#how-it-works)
*   [Key Concepts](#key-concepts)
    *   [Text Chunking](#text-chunking)
    *   [Sentence Embeddings](#sentence-embeddings)
    *   [Similarity Search](#similarity-search)
    *   [Augmented Prompting](#augmented-prompting)
*   [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Usage](#usage)
*   [Customization](#customization)
*   [Contributing](#contributing)
*   [License](#license)
*   [References](#references)

## Overview

Large Language Models (LLMs) have incredible generative capabilities but often lack knowledge of specific, private, or recent data. They can also be prone to "hallucination," generating plausible but incorrect information.

**Retrieval-Augmented Generation (RAG)** solves these problems by grounding the LLM in a set of external documents. Instead of just answering a question from its internal knowledge, the model first retrieves relevant information from your provided documents (e.g., a PDF textbook) and uses that context to generate a more accurate and factual answer.

This repository provides the complete code to build a RAG pipeline for chatting with a ~1200 page nutrition textbook, but it can be adapted for almost any PDF.

## Why Use RAG?

*   **Prevent Hallucinations**: By providing factual, retrieved information directly to the LLM in the prompt, RAG minimizes the chances of the model generating incorrect or made-up answers.
*   **Work with Custom Data**: RAG allows you to make an LLM an expert on your specific domain (e.g., medical journals, company documentation, email chains) without the need for expensive and time-consuming fine-tuning.
*   **Provide Verifiable Sources**: Because the answer is generated from specific retrieved passages, you can easily trace the source of the information, adding a layer of trust and interpretability.

## How It Works

The entire pipeline is broken down into two main stages:

**1. Document Preprocessing and Embedding (Indexing)**
This stage prepares your document for retrieval.
*   **Load Document**: Ingest a PDF file.
*   **Text Chunking**: The raw text is split into smaller, semantically meaningful chunks (e.g., groups of 5-10 sentences). This is crucial because embedding models have token limits and retrieval is more effective on focused text blocks.
*   **Generate Embeddings**: Each text chunk is converted into a numerical vector (an "embedding") using a sentence-transformer model like `all-mpnet-base-v2`. These embeddings capture the semantic meaning of the text.
*   **Store Embeddings**: The generated embeddings are stored in a file or a specialized vector database (like FAISS) for efficient searching.

**2. Search and Answer Generation (Retrieval & Generation)**
This stage handles the user interaction.
*   **User Query**: A user asks a question in natural language.
*   **Embed Query**: The same embedding model is used to convert the user's query into a vector.
*   **Similarity Search**: A vector search is performed between the query embedding and all the stored document chunk embeddings. This is typically done using **cosine similarity** to find the chunks that are most semantically related to the query.
*   **Augment Prompt**: The most relevant text chunks (the "context") are retrieved and inserted into a carefully crafted prompt for the LLM.
*   **Generate Answer**: The LLM receives the augmented prompt (containing the user's question and the retrieved context) and generates a final, context-aware answer.

## Key Concepts

### Text Chunking
Splitting large texts into smaller pieces is essential for RAG.
*   **Why?**
    *   To fit within the context window of the embedding model (e.g., 384 tokens for `all-mpnet-base-v2`).
    *   To ensure the retrieved context passed to the LLM is specific and focused, leading to better answers.
    *   To create more precise embeddings that aren't "diluted" by overly long, diverse text passages.
*   **Methods**: While simple rules (`.split(". ")`) work, NLP libraries like `spaCy` or `nltk` provide more robust sentence tokenization.

### Sentence Embeddings
We convert text into numbers so that a computer can understand its meaning.
*   **Model Choice**: We use `all-mpnet-base-v2` from the `sentence-transformers` library, which maps sentences and paragraphs to a 768-dimensional dense vector space. The [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is a great resource for comparing different embedding models.
*   **Fixed-Size Output**: Regardless of the input text length, the embedding vector's size is fixed (e.g., 768 dimensions). The model will pad or truncate the input to fit its required sequence length.

### Similarity Search
To find the most relevant document chunks for a given query, we compare their embedding vectors.
*   **Cosine Similarity vs. Dot Product**: Cosine similarity is generally preferred for text similarity because it measures the *direction* (angle) of the vectors, not their magnitude. This makes it robust to differences in sentence length, focusing purely on semantic closeness.
*   **Indexing for Speed**: For small datasets, an exhaustive search is fast. For larger datasets (100,000+ embeddings), creating an index with a library like **FAISS** (Facebook AI Similarity Search) is crucial for speeding up the search process.

### Augmented Prompting
The quality of the final answer heavily depends on how you structure the prompt for the LLM. This is a form of **prompt engineering**.
*   **Best Practices**:
    1.  **Give Clear Instructions**: Tell the model exactly what to do (e.g., "Answer the user's question based *only* on the provided context.").
    2.  **Provide Context**: Clearly separate the retrieved passages from the user's question.
    3.  **Give Examples (Few-shot)**: Show the model examples of good input/output pairs.
    4.  **Give Room to Think**: Encourage step-by-step reasoning by asking the model to "show its work" before giving the final answer.

## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch
*   GPU with at least 8GB VRAM (for running the LLM locally)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/dahiya-04/NutriChat.git
    cd NutriChat
    ```
2.  Install the required packages. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` should include `transformers`, `torch`, `sentence-transformers`, `bitsandbytes`, `accelerate`, and a PDF reader library like `pypdf`.*

### Usage
1.  **Add your PDF**: Place the PDF file you want to chat with into the `data/` directory.
2.  **Process and Embed**: Run the script to process the PDF, chunk the text, and create embeddings.
    ```bash
    python ingest.py --pdf_path data/your_textbook.pdf
    ```
    This will generate and save the embeddings and text chunks locally.
3.  **Chat with the PDF**: Start the interactive chat script.
    ```bash
    python chat.py
    ```
    The script will load the LLM and your processed data, and you can start asking questions!

## Customization

*   **LLM Model**: The `chat.py` script can be modified to use a different instruction-tuned LLM from Hugging Face. Remember to check the model's prompt template.
*   **Embedding Model**: You can swap `all-mpnet-base-v2` for any other model from the `sentence-transformers` library. Just ensure you use the same model for both indexing and querying.
*   **Chunking Strategy**: Experiment with different chunk sizes and sentence combinations in `ingest.py` to see how it affects retrieval quality.

## Contributing

Contributions are welcome! If you have suggestions for improvement or find any bugs, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

*   [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
*   [Sentence-Transformers Documentation](https://www.sbert.net/)
*   [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

[1] https://arxiv.org/abs/2005.11401
