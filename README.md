# Agentic-RAG

Retrieval-Augmented Generation (RAG) notebooks that answer questions from a long PDF book. The questions in `queries.json` are psychology topics, for example "What is the scientific method in psychology?", "What are the basic parts of a neuron?" and "What are the stages of sleep?".

## What is in this repo

| File | Purpose |
|---|---|
| `AgenticRAG_notebook.ipynb` | Full pipeline: extract text and section headings from the PDF, chunk it, build a FAISS index, retrieve context and generate answers with a local LLM |
| `rag-pipeline-for-qa.ipynb` | Simpler RAG pipeline that retrieves relevant pages and generates answers, with visualisations of the retrieved context (word cloud, plots) |
| `queries.json` | Questions to answer, each with a `query_id` and a `question` |

## How `AgenticRAG_notebook.ipynb` works

1. **Extract**: reads `book.pdf` with PyMuPDF (`fitz`) line by line and records the page number and the section a line belongs to (headings are detected with a regex such as `1.2 Title`).
2. **Chunk**: splits the text with LangChain's `RecursiveCharacterTextSplitter`, using the `thenlper/gte-large` tokenizer (chunk size 1500, overlap 500).
3. **Index**: embeds the chunks with `thenlper/gte-large` and stores them in a FAISS vector store with cosine distance.
4. **Retrieve**: a `RetrieverTool` runs a similarity search (default `k=7`). A targeted retrieval step also runs extra related queries for the scientific-method question and keeps matches from pages 40-60.
5. **Generate**: answers with the local `Qwen/Qwen2.5-3B-Instruct` model from Hugging Face Transformers, and returns the section and page references used.

## Tech stack

Python, PyMuPDF, LangChain, FAISS, sentence-transformers / Hugging Face embeddings, Transformers, PyTorch.

## Running it

The notebooks were written for Kaggle. They read the dataset from `/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG/` (`book.pdf` and `queries.json`), and that path appears in the code. To run them elsewhere, change the path to your own copy of the PDF.

```bash
pip install torch transformers langchain PyMuPDF faiss-cpu tqdm sentence-transformers
```

`rag-pipeline-for-qa.ipynb` installs `sentence-transformers faiss-cpu wordcloud transformers PyPDF2`.

A GPU is used when one is available. The code falls back to CPU otherwise.
