from dotenv import load_dotenv
import streamlit as st
import os
import speech_recognition as sr
from langchain import OpenAI
from audio_recorder_streamlit import audio_recorder
from streamlit_mic_recorder import speech_to_text
import requests
from bs4 import BeautifulSoup
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate

load_dotenv()
llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model_name="gpt-3.5-turbo-instruct",temperature=0.8)


st.set_page_config(page_title="Ninja Bot")

st.header("Ninja Bot")

if'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]

input=st.text_input("Input:",key="input")


text = speech_to_text(
    language='en',
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)

res=""
out=0

def scrape(topic):
    url= []
    search = topic
    results = 5  # valid options 10, 20, 30, 40, 50, and 100
    page = requests.get(f"https://www.google.com/search?q={search}&num={results}")
    soup = BeautifulSoup(page.content, "html5lib")
    links = soup.findAll("a")
    for link in links :
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            #print(link.get('href').split("?q=")[1].split("&sa=U")[0])
            url.append(link.get('href').split("?q=")[1].split("&sa=U")[0])

    url.pop(0)
    url.pop(0)
    url.pop(len(url)-1)
    url.pop(len(url)-1)

    content=[]
    for link in url:
        req=requests.get(link)
        soup= BeautifulSoup(req.content, "html.parser")
        text=soup.get_text()
        tex=text.replace(" ","")
        content.append(text)
        return text


def point_summerizer(text):
    llm.get_num_tokens(text)
    text_split=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=20)
    chunks=text_split.create_documents({text})
    chain=load_summarize_chain(
    llm,
    chain_type='map_reduce',
    verbose=False
    )
    summary=chain.run(chunks)
    return summary


if input:
    data=scrape(input)
    res=point_summerizer(data)
    st.session_state['chat_history'].append(("You",input))
    out=1
if text:
    data=scrape(text)
    res=point_summerizer(data)
    st.session_state['chat_history'].append(("You",text))
    out=1
if out==1:
    st.subheader("The response is")
    st.write(res)
    st.session_state['chat_history'].append(("Bot",res))
    out=0

st.subheader("The chat History")
for role,text in st.session_state['chat_history']:
    st.write(role,":",text)