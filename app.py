import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Intern Support AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

FAQ_FILE = Path(__file__).parent / "intern_faq.csv"

# Minimum similarity required to return an answer
MIN_SCORE = 0.30

WELCOME_MESSAGE = "Hi! 👋 Welcome to Intern Support. How can I help you today?"

NO_MATCH_MESSAGE = (
    "Sorry, I couldn't find a suitable answer. "
    "Please contact your internship supervisor for help."
)


# --------------------------------------------------
# 2. Load FAQ Dataset
# --------------------------------------------------

@st.cache_data
def load_faq():
    if not FAQ_FILE.exists():
        raise FileNotFoundError(
            "intern_faq.csv was not found. "
            "Place it in the same folder as this app file."
        )

    df = pd.read_csv(FAQ_FILE)

    required_columns = {"question", "answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError("The CSV must contain 'question' and 'answer' columns.")

    df = df.dropna(subset=["question", "answer"])

    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    df = df[(df["question"] != "") & (df["answer"] != "")]

    if df.empty:
        raise ValueError("The FAQ dataset is empty.")

    return df.reset_index(drop=True)


try:
    faq_df = load_faq()
except Exception as error:
    st.error(f"Could not load FAQ dataset: {error}")
    st.stop()


# --------------------------------------------------
# 3. Alternative Questions for Better Matching
# --------------------------------------------------
# The key is the row number in the CSV (starting from 0).
# The list holds other ways to ask the same question.

QUESTION_VARIATIONS = {
    0: [
        "What are the working hours?",
        "When do I need to work?",
        "What time does my internship start and finish?",
        "What is my internship schedule?",
    ],
    1: [
        "How can I turn in my assignment?",
        "Where should I submit my work?",
        "How do I hand in my task?",
        "How can I submit my completed assignment?",
    ],
    2: [
        "How can I obtain my certificate?",
        "How do I get my internship completion document?",
        "When can I get my internship completion certificate?",
        "How do I receive my internship certificate?",
        "How can I get my certificate after completing the internship?",
    ],
    3: [
        "How do I get in touch with my supervisor?",
        "Who should I contact for help?",
        "How can I reach my supervisor?",
    ],
    4: [
        "How can I receive feedback on my work?",
        "How do I get my project reviewed?",
        "Can my supervisor review my project?",
    ],
    5: [
        "What should I do if I miss the due date?",
        "What if my task deadline has passed?",
        "I missed my deadline. What should I do?",
    ],
    6: [
        "Is late task submission allowed?",
        "Can I hand in my assignment after the deadline?",
    ],
    7: [
        "How do I push my code to GitHub?",
        "How can I upload project files to GitHub?",
        "How do I submit code using GitHub?",
        "How do I put my code on GitHub?",
    ],
    8: [
        "Why do we use GitHub?",
        "What is the purpose of GitHub?",
        "Why is GitHub important for my internship?",
    ],
    9: [
        "How can I make a new GitHub repository?",
        "How do I start a repository on GitHub?",
    ],
    10: [
        "What should I do if my code is not working?",
        "I am having trouble with Python. What should I do?",
        "I have a coding error. How can I get help?",
        "My Python program is not running. What should I do?",
        "How do I troubleshoot a programming error?",
    ],
    11: [
        "Can I ask for help with my assignment?",
        "Where can I ask questions about my task?",
    ],
    12: [
        "How can I see my internship progress?",
        "How do I track my completed tasks?",
    ],
    13: [
        "What if I cannot finish my assignment?",
        "What happens if I fail to complete a task?",
    ],
    14: [
        "Can I use Visual Studio Code?",
        "Is VS Code allowed for internship projects?",
    ],
    15: [
        "Which version of Python do I need?",
        "What Python version is recommended?",
    ],
    16: [
        "How can I install a Python package?",
        "How do I install libraries using pip?",
        "How can I install a module in Python?",
    ],
    17: [
        "What files do I need to submit?",
        "What should my assignment contain?",
        "Which files should I include in my project?",
    ],
    18: [
        "How do I create project documentation?",
        "What goes into a README?",
        "How can I document my project?",
    ],
    19: [
        "Can I make changes after submitting my task?",
        "How do I correct my submitted assignment?",
    ],
    20: [
        "What if the submission website is not working?",
        "I cannot log in to the submission platform. What should I do?",
    ],
    21: [
        "Can I get more time to finish my task?",
        "How do I ask for extra time?",
        "I need an extension for my assignment.",
    ],
    22: [
        "Do I have to attend internship meetings?",
        "Are meetings mandatory?",
    ],
    23: [
        "How do I tell someone about a technical problem?",
        "Where should I report a technical issue?",
    ],
    24: [
        "My supervisor is not replying. What should I do?",
        "What if I do not receive a response from my supervisor?",
    ],
    25: [
        "Can I use internet tutorials to complete my task?",
        "Are online learning resources allowed?",
    ],
    26: [
        "How do I keep project information confidential?",
        "Can I share private project data online?",
    ],
    27: [
        "Can I include my internship work in my portfolio?",
        "Can I showcase my project on my portfolio?",
        "Can I add my internship project to my CV?",
    ],
    28: [
        "What should I do when all tasks are completed?",
        "What happens when I finish my internship assignments?",
    ],
    29: [
        "Am I eligible for an internship certificate?",
        "What are the requirements for receiving a certificate?",
        "Who can tell me if I qualify for a certificate?",
    ],
}


# --------------------------------------------------
# 4. Prepare Questions for Matching
# --------------------------------------------------

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Keep each original question and its alternative phrasings.
all_questions = []
question_owner = []

for index, row in faq_df.iterrows():
    variations = QUESTION_VARIATIONS.get(index, [])

    for question in [row["question"]] + variations:
        all_questions.append(normalize_text(question))
        question_owner.append(index)


# Stop words ("what", "is", "how", "do") are removed so they
# do not dominate the similarity score.
@st.cache_resource
def create_model(questions):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors


vectorizer, question_vectors = create_model(all_questions)


# --------------------------------------------------
# 5. Match User Question to an FAQ
# --------------------------------------------------

def get_answer(user_question):
    normalized_question = normalize_text(user_question)

    if not normalized_question:
        return "Please enter a question.", 0.0, False

    user_vector = vectorizer.transform([normalized_question])

    # No meaningful word found in the FAQ vocabulary: do not guess.
    if user_vector.nnz == 0:
        return NO_MATCH_MESSAGE, 0.0, False

    similarities = cosine_similarity(user_vector, question_vectors).flatten()

    best_match_position = int(similarities.argmax())
    best_score = float(similarities[best_match_position])

    if best_score < MIN_SCORE:
        return NO_MATCH_MESSAGE, best_score, False

    # Map the matched variation back to its original FAQ row.
    faq_index = question_owner[best_match_position]
    answer = faq_df.iloc[faq_index]["answer"]

    return answer, best_score, True


# --------------------------------------------------
# 6. Chatbot Interface
# --------------------------------------------------

st.title("🤖 Intern Support AI Chatbot")

st.write(
    "Your assistant for internship questions, task submissions, "
    "deadlines, GitHub, and technical support."
)

st.info(
    "Ask a question below. The chatbot searches the internship "
    "FAQ and recognizes common alternative phrasings."
)

with st.expander("💡 Example Questions"):
    st.markdown(
        """
- How do I submit my task?
- How do I upload my code to GitHub?
- How can I obtain my certificate?
- I am having trouble with Python. What should I do?
- How do I install Python libraries?
- How can I request an extension?
"""
    )


# --------------------------------------------------
# 7. Chat History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your internship question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    answer, score, matched = get_answer(user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if matched:
            st.caption(f"FAQ similarity score: {score:.2%}")


# --------------------------------------------------
# 8. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("📋 Chatbot Information")

    st.write(f"Available FAQs: {len(faq_df)}")
    st.write("NLP Model: TF-IDF")
    st.write("Matching: FAQ variations + cosine similarity")
    st.write("Interface: Streamlit")

    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
        st.rerun()