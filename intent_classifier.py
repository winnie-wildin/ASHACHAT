import google.generativeai as genai

def classify_intent(query: str) -> str:
    # Prompt Gemini to respond ONLY with one of: "job_search", "event_details", "general"
    prompt = f"""
    Classify the following user query into one of three categories:
    1. job_search (if they are looking for job listings, openings, or jobs)
    2. event_details (if they are asking about a specific event, its details, time, place, etc.)
    3. general (anything else)

    Only return the category name exactly (job_search, event_details, general).
    Query: "{query}"
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    # Clean response (just in case)
    answer = resp.text.strip().split('\n')[0].lower()
    # Make sure it's a valid choice
    if "job_search" in answer:
        return "job_search"
    elif "event_details" in answer:
        return "event_details"
    elif "general" in answer:
        return "general"
    else:
        return "general"