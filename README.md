# 🤖 Intern Support AI Chatbot

An AI-powered chatbot designed to help interns get quick answers to frequently asked questions about internships, assignments, supervisors, certificates, GitHub, and other internship-related activities.

The chatbot uses **TF-IDF vectorization and cosine similarity** to identify the most relevant FAQ and provide an appropriate response through an interactive Streamlit interface.

## 📌 Project Overview

Interns often have questions about internship procedures, task submission, deadlines, certificates, and communication with supervisors. Finding answers to these questions can take time.

The Intern Support AI Chatbot provides a simple and interactive solution by matching users' questions with a predefined FAQ dataset and returning relevant answers.

## 🎯 Objectives

* Provide quick answers to common internship-related questions.
* Reduce repetitive questions and improve access to information.
* Apply Natural Language Processing (NLP) techniques to question matching.
* Develop an interactive and user-friendly chatbot.
* Demonstrate practical AI and machine learning implementation.

## ✨ Features

* **Interactive Chat Interface:** Communicate with the chatbot through a Streamlit web application.
* **FAQ-Based Question Answering:** Retrieve answers from a predefined dataset.
* **Text Similarity Matching:** Use TF-IDF and cosine similarity to compare questions.
* **Alternative Question Phrasings:** Improve matching for different ways of asking similar questions.
* **Confidence Threshold:** Use a minimum similarity score to determine whether a question matches an FAQ.
* **Fallback Response:** Provide a default response when no suitable answer is found.
* **Chat History:** View previous messages during the session.
* **Clear Chat:** Start a fresh conversation when needed.

## 🛠️ Technologies Used

| Technology        | Purpose                                       |
| ----------------- | --------------------------------------------- |
| Python            | Core programming language                     |
| Streamlit         | Web interface                                 |
| Pandas            | Dataset loading and management                |
| Scikit-learn      | Text vectorization and similarity calculation |
| TF-IDF            | Convert text into numerical feature vectors   |
| Cosine Similarity | Measure similarity between questions          |
| CSV               | Store FAQ questions and answers               |

## 🧠 How It Works

1. The user enters a question in the chatbot interface.
2. The system prepares the FAQ questions and their alternative phrasings.
3. TF-IDF converts the text into numerical vectors.
4. Cosine similarity compares the user's question with the available FAQ questions.
5. The system identifies the best-matching question.
6. If the similarity score meets the configured threshold, the corresponding answer is displayed.
7. If no suitable match is found, the chatbot returns a fallback response.

**Matching threshold:** 0.30

## 📂 Project Structure

```text
Intern-Support-AI-Chatbot/
│
├── app.py
├── intern_faq.csv
├── requirements.txt
├── Intern_Support_AI_Chatbot_Report.pdf
├── README.md
└── .gitignore
```

**File descriptions:**

* `app.py` — Contains the chatbot application and question-matching logic.
* `intern_faq.csv` — Contains frequently asked questions and their answers.
* `requirements.txt` — Lists the required Python packages.
* `Intern_Support_AI_Chatbot_Report.pdf` — Provides the detailed project report.
* `README.md` — Documents the project and its setup instructions.
* `.gitignore` — Specifies files and folders that Git should ignore.

## ⚙️ Installation and Setup

### Prerequisites

* Python installed on your computer.
* Git (optional, for cloning the repository).

### Step 1: Clone the Repository

```bash
git clone https://github.com/alishbashaukat2-commits/Intern-Support-AI-Chatbot.git
```

### Step 2: Open the Project Folder

```bash
cd Intern-Support-AI-Chatbot
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the chatbot.

## 🧪 Testing

The chatbot was manually tested with different types of questions, including:

* Internship certificate requirements.
* Assignment submission and deadlines.
* Supervisor communication.
* GitHub usage and project uploads.
* Python troubleshooting.
* General questions unrelated to internships.

The tests showed that the chatbot can retrieve relevant answers for many supported questions and provide fallback responses for unrelated questions.

**Note:** The chatbot uses FAQ-based matching rather than a generative language model. Its responses depend on the questions and answers available in the dataset, and some differently phrased questions may produce incorrect matches.

## ⚠️ Limitations

* The chatbot can answer only questions covered by its FAQ dataset.
* It may return an incorrect answer when two questions have similar wording.
* It does not generate new answers independently.
* Its performance depends on the quality and coverage of the FAQ dataset.
* It does not currently connect to an external database or live internship management system.

## 🚀 Future Improvements

* Expand the FAQ dataset with more questions and answer variations.
* Improve question matching and reduce incorrect responses.
* Add multilingual support, including Urdu.
* Integrate a more advanced NLP model.
* Store chat history persistently.
* Add an administrator interface to manage FAQs.
* Deploy the chatbot online for wider accessibility.

## 👩‍💻 Author

**Alishba Shaukat**

BS Artificial Intelligence Student

GitHub: [alishbashaukat2-commits](https://github.com/alishbashaukat2-commits)

LinkedIn: [Alishba Shaukat](https://www.linkedin.com/in/alishba-shaukat-2a8b5442a/)

## 📄 Project Report

The detailed project report is available in the repository as `Intern_Support_AI_Chatbot_Report.pdf`.

---

*Developed as an AI project to demonstrate practical applications of NLP, text similarity, and interactive chatbot development.*
