from App.src.VectorStore.RetrieveVectorStoreIndex import Retrive_Vector_Store
import streamlit as st
from App.src.Chatbot.prompt import Custom_prmpt
from App.src.Chatbot.QueryEngine import Query_Engine
from App.client import create_client
import asyncio
import nest_asyncio
nest_asyncio.apply()


async def main():
    st.set_page_config(page_title="ChatBot")
    st.title("ChatBot")

    default_query_engine = await create_client()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if prompt := st.chat_input("write your message"):
        response = default_query_engine.query(prompt)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    asyncio.run(main())