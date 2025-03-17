from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
import os



def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    api_key=os.environ.get("GROQ_API_KEY")
    if api_key==None:
        llm = Ollama(model="phi3:latest", request_timeout=160.0)
    else:
        llm = Groq(model="llama3-70b-8192", api_key=api_key)
    

    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    embed_model = OllamaEmbedding(
        model_name="all-minilm:latest",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )

    vector_query_engine = vector_index.as_query_engine(llm=llm)
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        llm=llm,
        verbose=True
    )
    return query_engine