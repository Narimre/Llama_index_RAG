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
from utils import get_doc_tools
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner



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

vector_tool, summary_tool = get_doc_tools("/home/imolnar/Llama_index_RAG/Lesson3/data/metagpt.pdf", "metagpt",llm)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker, llm=llm)

response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

print(response.source_nodes[0].get_content(metadata_mode="all"))

response = agent.chat(
    "Tell me about the evaluation datasets used."
)

response = agent.chat("Tell me the results over one of the above datasets.")

############################################
###Lower-Level: Debuggability and Control
############################################

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker,llm=llm)


#create task and step by step run
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

step_output = agent.run_step(task.task_id)

completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)

step_output = agent.run_step(task.task_id) #injection
print(step_output.is_last)

response = agent.finalize_response(task.task_id)

print(str(response))


