import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk, HumanMessageChunk, SystemMessageChunk
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from CustomTools.run_shell_command import RunShellCommandTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

# Define tools
search = TavilySearchResults(max_results=3)
run_shell_command = RunShellCommandTool()

tools = [search, run_shell_command]

model = init_chat_model("gpt-4o-mini", model_provider="openai")

checkpointer = InMemorySaver()

colonel_agent = create_react_agent(model, tools, checkpointer=checkpointer)


# Agent-Loop Prompt

config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
# response = colonel_agent.invoke({"messages": [HumanMessage(content="")]}, config)
# print(response)

while True:
    user_prompt = input()
    response = colonel_agent.invoke({"messages": [HumanMessage(content=user_prompt)]}, config)
    print(response)
