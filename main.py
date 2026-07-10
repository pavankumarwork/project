import os
import json
from typing import List, Optional
import main  as st
from pydantic import BaseModel # type: ignore
from dotenv import load_dotenv # type: ignore
from google import genai
from google.genai import types # type: ignore

# ----------------------------------------------------------------------
# 1. INITIALIZATION & SETUP
# ----------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="EduGenie: Gemini Learning Assistant", 
    page_icon="🧞", 
    layout="wide"
)

# Initialize the official Google Gen AI Client
# Will read GEMINI_API_KEY from environment or integrate with Vertex AI
@st.cache_resource
def get_gemini_client():
    return genai.Client()

try:
    client = get_gemini_client()
    MODEL_ID = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Failed to initialize Google Gen AI Client. Verify API keys. Error: {e}")
    st.stop()

# ----------------------------------------------------------------------
# 2. CORE BACKEND AI LOGIC ENGINE (FastAPI/Pydantic Style Formats)
# ----------------------------------------------------------------------
class QuizRequest(BaseModel):
    topic: str
    difficulty: str
    num_questions: int

class SummaryRequest(BaseModel):
    text_content: str
    target_audience: str

def generate_structured_quiz(data: QuizRequest) -> str:
    """Generates a strictly structured JSON multiple-choice quiz using Gemini Schemas"""
    prompt = f"""
    You are an elite academic tutor. Create an original educational quiz about '{data.topic}' 
    tailored for a '{data.difficulty}' understanding level. Provide exactly {data.num_questions} questions.
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "id": types.Schema(type=types.Type.INTEGER),
                        "question": types.Schema(type=types.Type.STRING),
                        "options": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                        "correct_answer": types.Schema(type=types.Type.STRING),
                        "explanation": types.Schema(type=types.Type.STRING)
                    },
                    required=["id", "question", "options", "correct_answer", "explanation"]
                )
            )
        )
    )
    return response.text

def summarize_materials(data: SummaryRequest) -> str:
    """Transforms heavy reading text into organized study bullets and concepts"""
    prompt = f"""
    You are an advanced text synthesizer. Break down the following educational content into 
    key-takeaway bullet points, primary core definitions, and a high-yield summary.
    Adapt the language tone for a {data.target_audience}.
    
    Content to digest:
    {data.text_content}
    """
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text

def process_tutor_chat(history_list: list, new_message: str) -> str:
    """Processes multi-turn conversations using native SDK context states"""
    sdk_history = []
    for msg in history_list:
        sdk_history.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])]
            )
        )
    
    chat_session = client.chats.create(
        model=MODEL_ID,
        history=sdk_history,
        config=types.GenerateContentConfig(
            system_instruction="You are EduGenie, a supportive AI tutor. Guide students step-by-step instead of just giving answers instantly."
        )
    )
    response = chat_session.send_message(new_message)
    return response.text

# ----------------------------------------------------------------------
# 3. STREAMLIT FRONTEND USER INTERFACE
# ----------------------------------------------------------------------
st.title("🧞 EduGenie: Gemini-Powered Learning Assistant")
st.caption("Google Cloud Generative AI Group Project Architecture")

tab1, tab2, tab3 = st.tabs(["⚡ AI Quiz Generator", "📚 Study Summarizer", "💬 Companion Tutor Chat"])

# --- TAB 1: QUIZ GENERATOR UI ---
with tab1:
    st.header("Generate Tailored Assessment Quizzes")
    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.text_input("Enter Subject/Topic", placeholder="e.g., Photosynthesis, Cloud Computing")
    with col2:
        difficulty = st.selectbox("Complexity Level", ["Beginner", "Intermediate", "Advanced"])
    with col3:
        num_q = st.slider("Number of Questions", 1, 5, 3)
        
    if st.button("Build My Quiz", type="primary"):
        if topic:
            with st.spinner("Gemini is structuring your questions inside JSON schemas..."):
                try:
                    payload = QuizRequest(topic=topic, difficulty=difficulty, num_questions=num_q)
                    raw_json_str = generate_structured_quiz(payload)
                    st.session_state.current_quiz = json.loads(raw_json_str)
                except Exception as e:
                    st.error(f"Error compiling quiz: {e}")
        else:
            st.warning("Please specify a topic first.")

    if "current_quiz" in st.session_state:
        st.markdown("---")
        for idx, item in enumerate(st.session_state.current_quiz):
            st.markdown(f"**Q{idx+1}: {item['question']}**")
            for option in item['options']:
                st.write(f"- {option}")
            with st.expander("💡 Reveal Solution & Concept Framework"):
                st.success(f"Correct Answer: {item['correct_answer']}")
                st.info(item['explanation'])

# --- TAB 2: SUMMARIZER UI ---
with tab2:
    st.header("Convert Materials into Actionable Study Guides")
    doc_input = st.text_area("Paste Textbook Material / Raw Lecture Notes", height=200)
    audience = st.radio("Target Adaptivity Scale", ["High Schooler", "Undergrad Student", "Expert Specifier"], horizontal=True)
    
    if st.button("Synthesize Material"):
        if doc_input:
            with st.spinner("Tokenizing content and prompting structural overview..."):
                try:
                    payload = SummaryRequest(text_content=doc_input, target_audience=audience)
                    summary_result = summarize_materials(payload)
                    st.markdown("### 📋 EduGenie Smart Study Guide")
                    st.write(summary_result)
                except Exception as e:
                    st.error(f"Error parsing document: {e}")
        else:
            st.warning("Please paste valid text parameters first.")

# --- TAB 3: TUTOR CHAT UI ---
with tab3:
    st.header("Chat With EduGenie")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display active history dialogue lines
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    if user_input := st.chat_input("Ask a follow-up calculation or conceptual question..."):
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("model"):
            with st.spinner("EduGenie is thinking..."):
                try:
                    bot_reply = process_tutor_chat(st.session_state.chat_history, user_input)
                    st.write(bot_reply)
                    
                    # Persist turn snapshots within global session dictionaries
                    st.session_state.chat_history.append({"role": "user", "text": user_input})
                    st.session_state.chat_history.append({"role": "model", "text": bot_reply})
                except Exception as e:
                    st.error(f"Chat Context Session Error: {e}")