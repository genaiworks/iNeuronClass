import langgraph
import travel_agent_helper
import streamlit as st
import travel_agent_helper
from dotenv import load_dotenv
import streamlit as st 
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import uuid

def get_response(question: str, passenger_id:str):
    sm = travel_agent_helper.StateMachine()
    chain = sm.chain
    thread_id={"configurable": {"thread_id": str(uuid.uuid4())}}
    # st.session_state["result"]=st.session_state['dm'].start()
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": thread_id,
        }
    }
    event = chain.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    print("got response ******")
    print(event)

st.set_page_config(page_title="Chat with Database", page_icon=":speech_balloon:")
st.title("Human-In-The-Loop AI Collaboration with Reflection Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content= "Hello I am database assistant.  Ask me anything about your database."),
    ]


# st.session_state['dm'] = travel_agent_helper.StateMachine()
# st.session_state["result"]=st.session_state['dm'].start()

with st.sidebar:
    st.markdown("""
### What it's all about:

    This application demonstrates
    how artificial intelligence
    agents and a human (you) can
    collaborate on a task.
    
    Today's task is to write a news
    article about a meeting for 
    which a text transcript or 
    minutes are available.
    
    You point to that source;
    the writer agent drafts;
    the critique agent critiques;
    you can edit either the draft or
    the critique. This repeats until
    you are satisfied with a draft.
    v0.0.3
""")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

passenger_id = st.text_input("Passenger_id")
question = st.chat_input("Type here to chat with the database")

if question is not None:
    st.session_state.chat_history.append(HumanMessage(content=question))

    with st.chat_message("Human"):
        st.markdown(question)
            
    with st.chat_message("AI"):
        response = get_response(question, passenger_id)
        st.markdown(response)
    
    st.session_state.chat_history.append(AIMessage(content=response))