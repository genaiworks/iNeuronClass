import functools
import operator
import json
import os

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.vectorstores.redis import Redis
from langchain_experimental.tools import PythonREPLTool
from langchain_openai.chat_models import ChatOpenAI
from typing import TypedDict, Annotated, Sequence, Any
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langchain.tools import tool
from google.cloud import bigquery

#This is used for chat memory
memory = SqliteSaver.from_conn_string(":memory:")

os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds.json"

llm = ChatOpenAI(model="gpt-4-1106-preview", streaming=True)

repl = PythonREPLTool()

#This tool is used to get data from bigquery
@tool
def execute_sql_bigquery(sql):
  """Use this to execute sql query in BigQuery and get the result."""
  client = bigquery.Client()
  query_job = client.query(sql)
  results = query_job.result()
  result = list()

  for row in results:
      result.append(dict(row.items()))

  return result

#This python tool is used for visualisation
@tool
def python_repl(
      code: Annotated[str, "The python code to execute to generate your chart."]
):
  """Use this to execute python code. If you want to see the output of a value,
  you should print it out with `print(...)`. This is visible to the user."""
  try:
      result = repl.run(code)
  except BaseException as e:
      return f"Failed to execute. Error: {repr(e)}"
  return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

tools = [python_repl]


while True:

  #get input query from user
  quest = input("Please enter your question: ")

  matched_schema = get_matching_docs_from_vector(query=quest)
  
  # the text-to-sql,python node & conversation node are created using this function
  def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
      # Each worker node will be given a name and some tools.
      prompt = ChatPromptTemplate.from_messages(
          [
              (
                  "system",
                  system_prompt,
              ),
              MessagesPlaceholder(variable_name="messages"),
              MessagesPlaceholder(variable_name="agent_scratchpad"),
          ]
      ).partial(matched_schema=matched_schema)

      agent = create_openai_tools_agent(llm, tools, prompt)
      executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
      return executor

  def agent_node(state, agent, name):
      result = agent.invoke(state)
      return {"messages": [HumanMessage(content=result["output"], name=name)]}

  members = ["text_to_sql", "sql_to_python", "Conversation"]
  
  #this prompt is for supervisor
  system_prompt = (
      " You are a supervisor tasked with managing a conversation between the"
      " following workers:  {members}. Given the following user request,"
      " respond with the worker to act next. Each worker will perform a"
      " task and respond with their results."
      " If you or any of the other {members} have the final answer or deliverable"
      " prefix your response with FINISH: so you know it is time to stop. Once any of the {members} respond"
      " with a final answer or deliverable return the response to the user and stop the execution. "
      )


  options = ["FINISH"] + members
  function_def = {
      "name": "route",
      "description": "Select the next role.",
      "parameters": {
          "title": "routeSchema",
          "type": "object",
          "properties": {
              "next": {
                  "title": "Next",
                  "anyOf": [
                      {"enum": options},
                  ],
              }
          },
          "required": ["next"],
      },
  }
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          MessagesPlaceholder(variable_name="messages"),
          (
              "system",
              "Given the conversation above, who should act next?"
              " Or should we FINISH? Select one of: {options}. "
          ),
      ]
  ).partial(options=str(options), members=", ".join(members))

  supervisor_chain = (
          prompt
          | llm.bind_functions(functions=[function_def], function_call="route")
          | JsonOutputFunctionsParser()
  )

  class AgentState(TypedDict):
      # The annotation tells the graph that new messages will always
      # be added to the current states
      messages: Annotated[Sequence[BaseMessage], operator.add]
      # The 'next' field indicates where to route to next
      next: str

  
  #text to sql node is created , we pass bigquery tool so that it can get the data
  #from big query/sql
  text_to_sql_agent = create_agent(llm, [execute_sql_bigquery], text_to_sql_prompt)
  text_to_sql_node = functools.partial(agent_node, agent=text_to_sql_agent, name="text_to_sql")

  #python node is created and python repl tool is passed to execute visualisation
  sql_to_python_agent = create_agent(llm, [python_repl], sql_to_python_prompt)
  sql_to_python_node = functools.partial(agent_node, agent=sql_to_python_agent, name="sql_to_python")

  #this is conversation node
  conversation_agent = create_agent(llm,  [to_lower_case],
                                    " You will respond to user for general queries & conversations."
                                    " Always return the answer with "
                                    "prefix FINISH: when done.")

  conversation_node = functools.partial(agent_node, agent=conversation_agent, name="Conversation")

  #here we add nodes to the graph
  workflow = StateGraph(AgentState)
  workflow.add_node("text_to_sql", text_to_sql_node)
  workflow.add_node("sql_to_python", sql_to_python_node)
  workflow.add_node("Conversation", conversation_node)
  workflow.add_node("supervisor", supervisor_chain)

  for member in members:
      # We want our workers to ALWAYS "report back" to the supervisor when done
      workflow.add_edge(member, "supervisor")  # add one edge for each of the agents

  conditional_map = {k: k for k in members}
  conditional_map["FINISH"] = END

  #this is conditional edge, which means supervisor will decide wether to finish
  #or continue the conversation between nodes
  workflow.add_conditional_edges("supervisor", should_continue, conditional_map)
  
  #Finally, add entrypoint
  workflow.set_entry_point("supervisor")

  graph = workflow.compile(checkpointer=memory)

  #the thread id indicates the individual chat id, which is used to understand
  #chat context
  for event in graph.stream(
          {
              "messages": [
                  HumanMessage(content=quest)
              ]
          },{"configurable": {"thread_id": "2"}}
  ):
      for k, v in event.items():
          # if k != "__end__" and 'messages' in v:
          #     # print(v["messages"][0])
          #     print(v)
          if k != "__end__":
              print(v)

  # This will return the final message
  #print(event[END]['messages'][-1].content)