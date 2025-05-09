# job_api.py
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env at the root directory

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")


model = genai.GenerativeModel("gemini-2.0-flash")

def extract_job_title_gemini(user_input):
    prompt = f"""Extract the intended job title from this user query:\n"{user_input}"\nReturn only the job title, nothing else."""
    response = model.generate_content(prompt)
    return response.text.strip()

def fetch_real_time_jobs(query):
    url = "https://jsearch.p.rapidapi.com/search"
    
    cleaned_query = extract_job_title_gemini(query)
    
    querystring = {
        "query": f"{cleaned_query}",
        "page": "1",
        "num_pages": "1",
        "country": "us",
        "date_posted": "month"
    }
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def parse_api_results(api_json):
    """
    Takes results from fetch_real_time_jobs and returns a list of job dicts.
    """
    # Result schema depends on API - adjust as needed.
    if api_json.get("data"):  # This is the correct field for jSearch
        return api_json["data"]
    return []