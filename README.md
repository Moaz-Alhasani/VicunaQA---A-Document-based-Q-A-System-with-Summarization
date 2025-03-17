# VicunaQA---A-Document-based-Q-A-System-with-Summarization

# Description
VicunaQA is an advanced question-answering system built using LangChain, Hugging Face Transformers, and various NLP tools. The system processes and extracts information from documents fetched from URLs, splits them into chunks, and utilizes the Google Flan-T5 model for generating concise and meaningful answers to user queries.

This tool allows users to:

  Load documents from URLs.
  Split large documents into manageable chunks.
  Perform custom question-answering and summarization tasks using a transformer-based model.
  Retrieve relevant information from a vector database and present answers with sources.
# Features
  Document Loading: Load documents directly from URLs using the UnstructuredURLLoader.
  Text Chunking: Automatically split large documents into smaller chunks for more efficient processing.
  Summarization and Q&A: Use the Flan-T5 model to summarize documents and answer specific user queries.
  Vector Database: Integration with FAISS for fast, efficient document retrieval.
  Custom Model: A custom wrapper around Hugging Face’s Flan-T5 model to fine-tune and control output.
# Requirements
  Python 3.x
  pip install -r requirements.txt
  Core Libraries
  LangChain
  Hugging Face Transformers
  FAISS
  Sentence-Transformers
  PyPDF
  NLTK
# Installation
  Clone the repository:
  
  bash
  Copy
  Edit
  git clone https://github.com/your-username/VicunaQA.git
  cd VicunaQA
  Install the required libraries:

  bash
  Copy
  Edit
  pip install -r requirements.txt
  Download the necessary NLTK datasets:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Usage
Load your documents:

python
Copy
Edit
URLs = [
    'https://blog.gopenai.com/paper-review-llama-2-open-foundation-and-fine-tuned-chat-models-23e539522acb',
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]
loaders = UnstructuredURLLoader(urls=URLs)
data = loaders.load()
Split documents into chunks:

python
Copy
Edit
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
text_chunk = text_splitter.split_documents(data)
Set up the custom model and retrieval chain:

python
Copy
Edit
llm = LLMWapper(tokenizer=tokenizer, model=model, max_length=100)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstores.as_retriever())
Start querying:

python
Copy
Edit
while True:
    query = input("Enter your question: ")
    if query == 'exit':
        print('Exiting')
        break
    if query:
        result = chain({"question": query}, return_only_outputs=True)
        print(textwrap.fill(result['answer'], 100))
# License
This project is licensed under the MIT License - see the LICENSE file for details.
