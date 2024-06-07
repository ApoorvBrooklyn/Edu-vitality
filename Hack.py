import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load the trained model
model = load_model('learning_preference_model_rnn.h5')

# Define the label encoder for decoding the predicted labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Auditory Learner', 'Kinesthetic Learner', 'Logical Learner', 'Reading/Writing Learner', 'Social Learner', 'Solitary Learner', 'Visual Learner'])

# Function to predict learning preference
def predict_learning_preference(input_features):
    input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))
    prediction = model.predict(input_features)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    return predicted_label[0]

if 'current_stage' not in st.session_state:
    st.session_state['current_stage'] = "Introduction"


if 'selected_app' not in st.session_state:
    st.session_state['selected_app'] = None

# if st.session_state['selected_app'] is None:
#     st.title("Welcome to the App")
#     app_choice = st.selectbox("Choose an application", ["Quiz", "Chat"], key="app_choice_main")
#     if st.button("Proceed"):
#         st.session_state['selected_app'] = app_choice


def update_timer(new_section):
    if st.session_state.active_section != new_section:
        if st.session_state.section_start_time is not None:
            st.session_state.section_end_time = datetime.now()
            duration = (st.session_state.section_end_time - st.session_state.section_start_time).total_seconds()
            section = st.session_state.active_section
            if section in st.session_state.section_times:
                st.session_state.section_times[section] += duration
            else:
                st.session_state.section_times[section] = duration
        st.session_state.active_section = new_section
        st.session_state.section_start_time = datetime.now()
        st.session_state.section_end_time = None

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

if st.session_state['selected_app'] is None:
    st.title("Welcome to the App")
    app_choice = st.selectbox("Choose an application", ["Quiz", "Chat"])
    if st.button("Proceed"):
        st.session_state['selected_app'] = app_choice

# if st.session_state['selected_app'] is None:
#     st.title("Welcome to the App")
#     app_choice = st.selectbox("Choose an application", ["Quiz", "Chat"])
#     if st.button("Proceed"):
#         st.session_state['selected_app'] = app_choice

if st.session_state['selected_app'] == "Quiz":
    with st.sidebar:
        st.header("Stages")
        st.write("Welcome to the quiz! Please select a stage to begin.")
        selected_stage = st.radio("Select Stage", ["Introduction", "Stage1", "Stage2", "Stage3", "Result"], key="stage_radio", index=["Introduction", "Stage1", "Stage2", "Stage3", "Result"].index(st.session_state['current_stage']), disabled=True)

    if selected_stage != st.session_state['current_stage']:
        update_timer(selected_stage)
        st.session_state['current_stage'] = selected_stage

    if selected_stage != "Introduction":
        st.session_state['quiz_start_time'] = datetime.now() if st.session_state.quiz_start_time is None else st.session_state.quiz_start_time

    time_elapsed = datetime.now() - st.session_state.quiz_start_time if st.session_state.quiz_start_time else timedelta(0)
    time_remaining = st.session_state.quiz_duration - time_elapsed if st.session_state.quiz_start_time else st.session_state.quiz_duration

    if selected_stage == "Introduction":
        st.write("""
            ## Welcome to the Quiz!
            This quiz consists of multiple stages. Each stage has its own set of questions that you need to answer.
            You can switch between the stages using the sidebar. Your progress and time spent on each stage will be tracked.
            Enjoy the quiz and good luck!
        """)
        if st.button("Start Quiz"):
            selected_stage = "Stage1"
            st.session_state['current_stage'] = selected_stage
            update_timer(selected_stage)

    if selected_stage == "Stage1":
        st.audio("indian_heritage.mp3", start_time=0)
        questions = [
            {
                "question": "What is Diwali also known as?",
                "options": ["", "The Festival of Colors", "The Festival of Lights", "The Festival of Flowers", "The Festival of Music"],
                "correct_answer": "The Festival of Lights"
            },
            {
                "question": "What do people do during Diwali?",
                "options": ["", "Decorate their homes with oil lamps, candles, and colorful rangoli patterns", "Throw colored powders and water at each other", "Fast and pray"],
                "correct_answer": "Decorate their homes with oil lamps, candles, and colorful rangoli patterns"
            },
            {
                "question": "What is the significance of Diwali?",
                "options": ["", "It signifies the victory of light over darkness and good over evil", "It celebrates the harvest season", "It has replaced all other cricket tournaments"],
                "correct_answer": "It signifies the victory of light over darkness and good over evil"
            },
            {
                "question": "Why might cricket be considered a unifying factor in India?",
                "options": ["", "Because only a few people watch it", "Because it brings together people from different backgrounds", "Because it is only played in schools", "Because it is not well-known outside of India"],
                "correct_answer": "Because it brings together people from different backgrounds"
            },
        ]
        for i, q in enumerate(questions, start=1):
            st.header(f"Question {i}")
            st.write(q["question"])
            user_choice = st.radio("Choose an answer:", options=q["options"], key=f"stage1_q{i}", index=q["options"].index(st.session_state['stage1_answers'][i-1]) if st.session_state['stage1_answers'][i-1] else 0, disabled=bool(st.session_state['stage1_answers'][i-1]))
            if user_choice and not st.session_state['stage1_answers'][i-1]:
                st.session_state['stage1_answers'][i-1] = user_choice
                if user_choice == q["correct_answer"]:
                    st.success("Correct!")
                    st.session_state['stage1_score'] += 1
                else:
                    st.error("Incorrect. The correct answer is: " + q["correct_answer"])

        if all(st.session_state['stage1_answers']):
            if st.button("Next Stage"):
                selected_stage = "Stage2"
                st.session_state['current_stage'] = selected_stage
                update_timer(selected_stage)

        st.write("You Scored:", st.session_state['stage1_score'])

    if selected_stage == "Stage2":
        st.write("In India, cricket is more than just a sport; it's a passion that unites people from all walks of life. From children playing in the streets to professional matches in grand stadiums, cricket is everywhere. The Indian Premier League, commonly known as the IPL, has further boosted the popularity of cricket, bringing together international players and creating an exciting atmosphere. Every year, during the IPL season, families gather around their television sets to cheer for their favorite teams. Cricket legends like Sachin Tendulkar and Virat Kohli are celebrated heroes, inspiring young aspiring cricketers across the country. The sport has a rich history in India, with memorable moments such as India's victory in the 1983 World Cup and the 2011 World Cup, which brought immense pride and joy to the nation.")
        questions = [
            {
                "question": "What does IPL stand for?",
                "options": ["", "Indian Professional League", "International Premier League", "Indian Premier League", "Indian Player League"],
                "correct_answer": "Indian Premier League"
            },
            {
                "question": "Who are some of the cricket legends mentioned in the above paragraph?",
                "options": ["", "MS Dhoni and Rohit Sharma", "Sachin Tendulkar and Virat Kohli", "Kapil Dev and Anil Kumble", "Sourav Ganguly and Rahul Dravid"],
                "correct_answer": "Sachin Tendulkar and Virat Kohli"
            },
            {
                "question": "What impact does the IPL have on cricket in India?",
                "options": ["", "It has decreased the popularity of cricket", "It has brought together international players and created an exciting atmosphere", "It has made cricket less accessible to young players", "It has replaced all other cricket tournaments"],
                "correct_answer": "It has brought together international players and created an exciting atmosphere"
            },
            {
                "question": "Why might cricket be considered a unifying factor in India?",
                "options": ["", "Because only a few people watch it", "Because it brings together people from different backgrounds", "Because it is only played in schools", "Because it is not well-known outside of India"],
                "correct_answer": "Because it brings together people from different backgrounds"
            },
        ]
        for i, q in enumerate(questions, start=1):
            st.header(f"Question {i}")
            st.write(q["question"])
            user_choice = st.radio("Choose an answer:", options=q["options"], key=f"stage2_q{i}", index=q["options"].index(st.session_state['stage2_answers'][i-1]) if st.session_state['stage2_answers'][i-1] else 0, disabled=bool(st.session_state['stage2_answers'][i-1]))
            if user_choice and not st.session_state['stage2_answers'][i-1]:
                st.session_state['stage2_answers'][i-1] = user_choice
                if user_choice == q["correct_answer"]:
                    st.success("Correct!")
                    st.session_state['stage2_score'] += 1
                else:
                    st.error("Incorrect. The correct answer is: " + q["correct_answer"])

        if all(st.session_state['stage2_answers']):
            if st.button("Next Stage"):
                selected_stage = "Stage3"
                st.session_state['current_stage'] = selected_stage
                update_timer(selected_stage)

        st.write("You Scored:", st.session_state['stage2_score'])

    if selected_stage == "Stage3":
        img = Image.open("museum.jpg")
        st.image(img,caption="Describe",width=300)
        questions = [
            {
                "question": "The place in the picture is most likely a:",
                "options": ["", "library", "museum", "zoo", "playground"],
                "correct_answer": "museum"
            },
            {
                "question": "People in the picture are looking at:",
                "options": ["", "animals", "paintings and sculptures", "a movie", "books"],
                "correct_answer": "paintings and sculptures"
            },
            {
                "question": "The objects on display in the museum are most likely:",
                "options": ["", "photographs", "paintings and sculptures", "microscopes", "computers"],
                "correct_answer": "paintings and sculptures"
            },
            {
                "question": "Which of the following would you NOT expect to find in a museum?",
                "options": ["", "paintings", "microscopes", "sculptures", "ancient pottery"],
                "correct_answer": "microscopes"
            },
        ]
        for i, q in enumerate(questions, start=1):
            st.header(f"Question {i}")
            st.write(q["question"])
            user_choice = st.radio("Choose an answer:", options=q["options"], key=f"stage3_q{i}", index=q["options"].index(st.session_state['stage3_answers'][i-1]) if st.session_state['stage3_answers'][i-1] else 0, disabled=bool(st.session_state['stage3_answers'][i-1]))
            if user_choice and not st.session_state['stage3_answers'][i-1]:
                st.session_state['stage3_answers'][i-1] = user_choice
                if user_choice == q["correct_answer"]:
                    st.success("Correct!")
                    st.session_state['stage3_score'] += 1
                else:
                    st.error("Incorrect. The correct answer is: " + q["correct_answer"])

        if all(st.session_state['stage3_answers']):
            if st.button("Next Stage"):
                selected_stage = "Result"
                st.session_state['current_stage'] = selected_stage
                update_timer(selected_stage)

        st.write("You Scored:", st.session_state['stage3_score'])


    if selected_stage == "Result":
        st.write("## Quiz Results")
        st.write(f"Stage 1 Score: {st.session_state['stage1_score']}")
        st.write(f"Stage 2 Score: {st.session_state['stage2_score']}")
        st.write(f"Stage 3 Score: {st.session_state['stage3_score']}")

        total_score = st.session_state['stage1_score'] + st.session_state['stage2_score'] + st.session_state['stage3_score']
        st.write(f"Total Score: {total_score}")

        # Collect section times and scores for prediction
        section_times = st.session_state['section_times']
        input_features = np.array([
            section_times.get("Stage1", 0),
            section_times.get("Stage2", 0),
            section_times.get("Stage3", 0),
            st.session_state['stage1_score'],
            st.session_state['stage2_score'],
            st.session_state['stage3_score']
        ]).reshape(1, -1)

        predicted_preference = predict_learning_preference(input_features)
        st.write(f"Predicted Learning Preference: {predicted_preference}")

if st.session_state['selected_app'] == "Chat":
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    #st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
