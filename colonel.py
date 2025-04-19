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

agent_action_log_file = "~/Desktop/action_log.txt"


# Agent-Loop Prompt

config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
system_message = f"""
    You are a helpful assistant called colonel. You have access to the user's operating
    system and can run commands to do what the user asks. Do not run any dangerous commands 
    or commands that require sudo. You also have to log your actions in a log file located at
    {agent_action_log_file}.
"""

while True:
    user_prompt = input()
    response = colonel_agent.invoke({"messages": [HumanMessage(content=user_prompt),
                                                  SystemMessage(content=system_message)
                                                  ]}, config)
    print(response["messages"][-1].content)
