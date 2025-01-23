import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

db_path = "db"

@st.cache_resource
def get_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def load_llm(repoidhugg):
    llm = HuggingFaceEndpoint(
        repo_id=repoidhugg,
        temperature=0.5,
        model_kwargs={
            "token": "",
            "max_tokens": 512,
        },
    )
    return llm

def set_prompt(prompt):
    prompt_template = PromptTemplate(template=prompt, input_variables=["context", "question"])
    return prompt_template

prompt = """
use the context that has been provided for user's question.
if you dont know the answer just say i dont know dont make up answer.
dont provied anything out of the given context
Context:{context}
Question:{question}

just answer directly and if the user is trying to say greeting then say hi
"""
prompt_template = set_prompt(prompt)
repoidhugg = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    st.title("Ask Medical Chatbot")
    st.write("Source: The GALE ENCYCLOPEDIA of MEDICINE SECOND")
   
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_input = st.chat_input("Your question")
    if user_input:
        st.chat_message("User").markdown(user_input)
        st.session_state.messages.append({"role": "User", "content": user_input})

        db = get_db()
        llm = load_llm(repoidhugg)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

        response = qa.invoke(user_input)["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
