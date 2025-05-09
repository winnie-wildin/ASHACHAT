import streamlit as st
import google.generativeai as genai
from rag_job import load_faiss_db, rag
from intent_classifier import classify_intent  # or define in this file
from rag_event import load_faiss_db_events, event_rag
# --- Session state as before (history, user_input, input clearing flag)
from job_api import fetch_real_time_jobs, parse_api_results




def guardrail_check(query):
    """
    Returns a label: 'discouragement', 'bias', 'inappropriate', 'profanity', or None
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f'''
You are a moderation assistant for a women's community chatbot.
Classify the input into one of the following (just return the word):
- profanity (if the query contains profane or hateful language)
- bias (if the query shows gender, racial, or other discriminatory bias)
- discouragement (if the query shows extreme self-doubt, discouraging language, or lack of self-worth)
- ok (if the query is just a normal question or positive/supportive)
User query: "{query}"
Category:
'''
    label = model.generate_content(prompt).text.strip().lower()
    if label != "ok":
        return label
    return None

def get_encouragement_story():
    stories = [
    "Remember, every great achievement starts with the decision to try. Malala Yousafzai, once just a young girl facing incredible adversity, chose to speak up for her right to learn. Her courage not only changed her own life but inspired millions of others. Like Malala, you have the power to take the first step, no matter how difficult it looks. Every step you take opens up new opportunities for growth, learning, and purpose.",

    "It's natural to feel discouraged at times—especially when breaking into a new field or returning to work. Consider the story of Indra Nooyi, who began her journey as a young woman from India and ultimately became CEO of one of the biggest companies in the world, PepsiCo. She once said, 'When you assume negative intent, you’re angry. If you take away that anger and assume positive intent, you will be amazed.' Overcoming obstacles not only strengthens your skills but also shapes your courage. Stay positive and believe that your breakthrough is on its way!",

    "If you're doubting yourself, know that you are not alone. Many women have stood where you stand. J.K. Rowling, before becoming one of the world's best-selling authors, was rejected by twelve publishers. Imagine if she had given up after the first five, or ten! Your unique talents and voice are valuable. Every rejection or setback is a nudge closer to the opportunity that's meant for you. Keep moving forward and your moment will come.",

    "There’s an African proverb: 'If you want to go fast, go alone. If you want to go far, go together.' Building your career is a journey, and it’s okay to seek help, support, or mentorship along the way. Countless successful women—like Michelle Obama, Chhavi Rajawat, and Reshma Saujani—credit their growth to communities and networks. Don’t hesitate to reach out, ask questions, or join groups. Your support system can make your dreams not just possible, but inevitable.",

    "Picture this: Arunima Sinha lost her leg in a train accident, yet she didn’t let it stop her. She became the first female amputee to climb Mount Everest. Her story teaches us that what seems impossible can be achieved with perseverance and faith in yourself. Whenever you face an obstacle, remember that you have untapped strength within you. Give yourself permission to fail, to try again, and to shine.",

    "Every journey has its setbacks. Consider Oprah Winfrey—she faced so many challenges early in her career, but she persisted and eventually became a global icon, inspiring millions. She often says, 'Turn your wounds into wisdom.' Whatever you’re feeling now—be it self-doubt or fear—can be transformed into motivation. Let your unique story, with all its ups and downs, become your strength.",
    
    "When you think it’s too late to try, think of Colonel Harland Sanders—he was 65 when he began franchising his first KFC. It’s never too late to start something new, gain a new skill, or follow your true calling. Age, background, and setbacks do not define your future—your courage and resilience do."
]
    import random
    return random.choice(stories)

def get_dynamic_encouragement(query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "A user has submitted a message to a women's chatbot "
        "that contains inappropriate, biased, or disrespectful language. "
        "Reply with a warm, encouraging, and constructive message that gently reminds "
        "the user about the values of kindness, inclusivity, and respect. "
        "Inspire positivity and, if possible, share a quote or a story about the power of "
        "support among women. Do not repeat the user's words or reprimand aggressively, "
        "but focus on uplifting, redirecting, and inspiring.\n\n"

        f"User's message: \"{query}\"\n"
        "Your response:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.clear_input = False

st.title("ASHA AI BOT")

# ==== HISTORY UTILITY FUNCTION ====
def get_chat_history(n_turns=3):
    # Only grab the turns leading UP TO the current user input
    full = ""
    for chat in st.session_state.history[-n_turns:]:
        full += f"User: {chat['user']}\n"
        full += f"Bot: {chat['bot']}\n"
    return full

# === Display chat history in colored bubbles ===
for chat in st.session_state.history:
    st.markdown(
        f'<div style="background:#ADD8E6;color:#222;border-radius:8px;padding:10px;margin-bottom:5px;max-width:80%;margin-left:auto;"><b>User:</b> {chat["user"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#F0F0F0;color:#222;border-radius:8px;padding:10px;margin-bottom:10px;max-width:80%;"><b>Bot:</b> {chat["bot"]}</div>',
        unsafe_allow_html=True,
    )

with st.form("chat_form"):
    user_input = st.text_input(
        "Enter your question:",
        value=st.session_state.user_input,
        key="user_input",
        placeholder="Ask your job search question or event info...",
    )
    submitted = st.form_submit_button("Send")


def render_job_results(jobs):
    if not jobs:
        st.info("No real-time jobs found.")
        return
    for job in jobs:
        st.markdown(
            f"**{job.get('job_title', 'Job opening')}** at {job.get('employer_name', 'Unknown')}\n\n"
            f"Location: {job.get('job_city', 'N/A')}, {job.get('job_state', '')}\n\n"
            f"Description: {job.get('job_description', '')[:300]}..."
        )
        st.markdown("---")

if submitted and st.session_state.user_input:
    query = st.session_state.user_input
    query = st.session_state.user_input
    guardrail_label = guardrail_check(query)

    if guardrail_label:
        if guardrail_label == "discouragement":
            output = get_encouragement_story()
        elif guardrail_label in ["bias", "profanity", "inappropriate"]:  # Add 'inappropriate' if needed
            output = get_dynamic_encouragement(query) 
        else:
            output = get_encouragement_story()  # Fallback to encouragement

        st.session_state.history.append({"user": query, "bot": output})
        st.session_state.clear_input = True
        st.rerun()
# (Else: proceed with intent/routing, as before.)
    chat_history = get_chat_history(n_turns=3)
    intent = classify_intent(query)
    output = ""
    print(f"Intent classified as: {intent}")
    if intent == "job_search":
        # 1. Try RapidAPI first!
        api_json = fetch_real_time_jobs(query)
        jobs = parse_api_results(api_json)
        if jobs:  # Found jobs, render them and summarize for chat
            job_list = jobs[:5]  # Show top 5
            output = f"### Here are the top {len(job_list)} live jobs I found for you:\n"
            for i, job in enumerate(job_list, start=1):
                output += (
                    f"\n**{i}. {job.get('job_title', 'Job Title Unknown')}**\n"
                    f"- **Company:** {job.get('employer_name', 'Unknown')}\n"
                    f"- **Location:** {job.get('job_city', 'N/A')}, {job.get('job_state', '')}, {job.get('job_country', '')}\n"
                    f"- **Posted:** {job.get('job_posted_at_datetime_utc', 'Unknown date')}\n"
                    f"- **Short desc:** {job.get('job_description', '').strip()[:180]}...\n"
                    f"{f'- [Apply Here]({job.get("job_apply_link")})' if job.get('job_apply_link') else ''}\n"
                )
            st.session_state.history.append({"user": query, "bot": output})
            st.session_state.clear_input = True
            st.rerun()
        else:
            # 2. Fallback to RAG!
            db_faiss = load_faiss_db()
            output = rag(db_faiss, query, chat_history)
            st.session_state.history.append({"user": query, "bot": output})
            st.session_state.clear_input = True
            st.rerun()
    elif intent == "event_details":
        db_events = load_faiss_db_events()
        output = event_rag(db_events, query, chat_history)  # Use the event RAG function    
        print(f"Event RAG output: {output}")
    else:
        # Use Gemini as a generic LLM for general queries, including history
        model = genai.GenerativeModel("gemini-2.0-flash")
        general_prompt = (
    f"You are a helpful, encouraging career assistant for women.\n"
    f"You respond directly to the user, offering practical advice and support in a conversational and empathetic tone.\n"
    f"Here is our recent conversation:\n{chat_history}\n\n"
    f"User: {query}\n"
    f"Assistant:"
)

        result = model.generate_content(general_prompt)
        output = result.text

    st.session_state.history.append({"user": query, "bot": output})
    st.session_state.clear_input = True
    st.rerun()

if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.user_input = ""
    st.rerun()