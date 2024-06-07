import streamlit as st



st.title("📓 QUIZ")
st.subheader("Questions")

# Define multiple sets of questions
question_sets = [
    [
        {"question": "Choose an answer for question 1.1", "options": ["Option 1.1.1", "Option 1.1.2", "Option 1.1.3"]},
        {"question": "Choose an answer for question 1.2", "options": ["Option 1.2.1", "Option 1.2.2", "Option 1.2.3"]},
        {"question": "Choose an answer for question 1.3", "options": ["Option 1.3.1", "Option 1.3.2", "Option 1.3.3"]}
    ],
    [
        {"question": "Choose an answer for question 2.1", "options": ["Option 2.1.1", "Option 2.1.2", "Option 2.1.3"]},
        {"question": "Choose an answer for question 2.2", "options": ["Option 2.2.1", "Option 2.2.2", "Option 2.2.3"]},
        {"question": "Choose an answer for question 2.3", "options": ["Option 2.3.1", "Option 2.3.2", "Option 2.3.3"]}
    ]
]


if 'current_question_set' not in st.session_state:
    st.session_state.current_question_set = 0


questions = question_sets[st.session_state.current_question_set]


form = st.form(key="quiz_form")


answers = {}


for i, q in enumerate(questions):
    answers[f"q{i+1}"] = form.radio(q["question"], options=q["options"])

submitted = form.form_submit_button("Submit your answers")

if submitted:
    for i, q in enumerate(questions):
        st.write(f"Your answer for question {i+1}: {answers[f'q{i+1}']}")
    
    
    if st.session_state.current_question_set < len(question_sets) - 1:
        st.session_state.current_question_set += 1
    else:
        st.write("No more questions available.")
