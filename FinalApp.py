import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

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

# Initialize scores and state variables
if 'section_start_time' not in st.session_state:
    st.session_state['section_start_time'] = None
if 'section_end_time' not in st.session_state:
    st.session_state['section_end_time'] = None
if 'active_section' not in st.session_state:
    st.session_state['active_section'] = ''
if 'section_times' not in st.session_state:
    st.session_state['section_times'] = {}
if 'quiz_start_time' not in st.session_state:
    st.session_state['quiz_start_time'] = None
if 'quiz_duration' not in st.session_state:
    st.session_state['quiz_duration'] = timedelta(minutes=5)
if 'current_stage' not in st.session_state:
    st.session_state['current_stage'] = "Introduction"
if 'stage1_score' not in st.session_state:
    st.session_state['stage1_score'] = 0
if 'stage2_score' not in st.session_state:
    st.session_state['stage2_score'] = 0
if 'stage3_score' not in st.session_state:
    st.session_state['stage3_score'] = 0
if 'stage1_answers' not in st.session_state:
    st.session_state['stage1_answers'] = [None] * 4
if 'stage2_answers' not in st.session_state:
    st.session_state['stage2_answers'] = [None] * 4
if 'stage3_answers' not in st.session_state:
    st.session_state['stage3_answers'] = [None] * 4

# Function to update timer
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