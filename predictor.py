import os
import json
import re
import requests
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()

CURRICULUM_PATH = os.path.join(os.path.dirname(__file__), "data", "curriculum.json")

# Standard English stop words to filter out noise
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself"
}

def load_curriculum():
    """Loads subjects and bloom levels from JSON."""
    try:
        with open(CURRICULUM_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading curriculum: {e}")
        return {"subjects": {}, "bloom_levels": {}}

CURRICULUM = load_curriculum()


def predict_rule_based(question, subject_name):
    """
    Enhanced Offline NLP Matcher:
    - High priority to leading imperative verbs (e.g. Design, Calculate, Explain, Compare).
    - Stopword-filtered semantic keyword overlap with official Course Outcomes.
    """
    q_lower = question.lower()
    words = re.findall(r'\b[a-zA-Z]{2,}\b', q_lower)
    meaningful_words = [w for w in words if w not in STOP_WORDS]
    
    bloom_levels = CURRICULUM.get("bloom_levels", {})
    detected_bl = None

    # Step 1: Check leading verb (first 3 words usually contain the Bloom directive)
    leading_words = words[:3]
    for w in leading_words:
        for bl_code, bl_data in bloom_levels.items():
            if w in bl_data.get("verbs", []):
                detected_bl = bl_code
                break
        if detected_bl:
            break

    # Step 2: If leading verb not found, search entire question from higher cognitive order (L6 down to L1)
    if not detected_bl:
        for bl_code in ["L6", "L5", "L4", "L3", "L2", "L1"]:
            verbs = bloom_levels.get(bl_code, {}).get("verbs", [])
            if any(re.search(r'\b' + re.escape(v) + r'\b', q_lower) for v in verbs):
                detected_bl = bl_code
                break

    if not detected_bl:
        detected_bl = "L2"  # fallback

    bl_info = bloom_levels.get(detected_bl, {"name": "Understand", "description": ""})

    # Step 3: Match Course Outcome (CO) using keyword overlap scoring
    subject_data = CURRICULUM.get("subjects", {}).get(subject_name, {})
    cos = subject_data.get("cos", {})

    best_co = list(cos.keys())[0] if cos else "CO1"
    highest_score = -1

    for co_id, co_desc in cos.items():
        co_words = [w for w in re.findall(r'\b[a-zA-Z]{2,}\b', co_desc.lower()) if w not in STOP_WORDS]
        # Count overlapping domain keywords
        overlap_count = sum(1 for w in meaningful_words if w in co_words)
        
        # Exact bigram / concept bonus
        for i in range(len(meaningful_words) - 1):
            phrase = f"{meaningful_words[i]} {meaningful_words[i+1]}"
            if phrase in co_desc.lower():
                overlap_count += 3

        if overlap_count > highest_score:
            highest_score = overlap_count
            best_co = co_id

    co_desc = cos.get(best_co, "General Course Outcome")

    return {
        "co": best_co,
        "co_desc": co_desc,
        "bl": detected_bl,
        "bl_name": bl_info.get("name", "Understand"),
        "bl_desc": bl_info.get("description", "Cognitive skill level"),
        "source": "Smart NLP Matcher",
        "explanation": f"Matched cognitive action verb to '{detected_bl}: {bl_info.get('name')}' and domain keywords to '{best_co}'."
    }


def predict_with_openrouter(question, subject_name, api_key):
    """
    Calls OpenRouter LLM with active free models.
    Falls back gracefully if an endpoint is busy or rate-limited.
    """
    subject_data = CURRICULUM.get("subjects", {}).get(subject_name, {})
    cos = subject_data.get("cos", {})
    co_context = "\n".join([f"- {k}: {v}" for k, v in cos.items()])

    prompt = f"""You are an expert academic evaluator strictly classifying examination questions according to NBA/NAAC Course Outcomes (CO) and Bloom's Revised Taxonomy (BL).

Subject: {subject_name} (Code: {subject_data.get('code', 'N/A')})

Approved Course Outcomes for this subject:
{co_context}

Bloom's Taxonomy Levels:
- L1: Remember (recall facts, definitions, state, list)
- L2: Understand (explain concepts, describe, discuss, classify)
- L3: Apply (calculate, solve, implement, determine)
- L4: Analyze (differentiate, examine, compare, contrast)
- L5: Evaluate (justify, critique, assess, verify)
- L6: Create (design, formulate, synthesize, develop)

Examination Question:
"{question}"

You MUST respond ONLY with a valid JSON object in this exact format, with NO markdown backticks or commentary:
{{
  "co": "CO2",
  "bl": "L2",
  "bl_name": "Understand",
  "explanation": "Brief 1-2 sentence justification for this classification."
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yashkhandelwal16/CO-BL-Predictor",
        "X-Title": "CO-BL Predictor"
    }

    # Currently active and supported models on OpenRouter
    models = [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3.5-lightning:free",
        "liquid/lfm-2.5-2.6b:free"
    ]

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"].strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    co_id = parsed.get("co", "CO1")
                    # Ensure valid CO key
                    if co_id not in cos and co_id.replace("L", "") in cos:
                        co_id = co_id.replace("L", "")
                    bl_code = parsed.get("bl", "L2")
                    bl_info = CURRICULUM.get("bloom_levels", {}).get(bl_code, {})
                    return {
                        "co": co_id,
                        "co_desc": cos.get(co_id, ""),
                        "bl": bl_code,
                        "bl_name": parsed.get("bl_name", bl_info.get("name", "Understand")),
                        "bl_desc": bl_info.get("description", "Cognitive understanding & classification."),
                        "source": f"OpenRouter ({model.split('/')[1]})",
                        "explanation": parsed.get("explanation", "Classified using LLM.")
                    }
        except Exception as e:
            continue

    # Graceful fallback to Smart NLP if API calls are rate-limited
    result = predict_rule_based(question, subject_name)
    result["source"] = "Smart NLP (Offline / Fast)"
    return result


def predict(question, subject_name):
    """Main entry point for prediction."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return predict_with_openrouter(question, subject_name, api_key)
    else:
        return predict_rule_based(question, subject_name)


if __name__ == "__main__":
    test_q = "Explain the radiation mechanism of a half-wave dipole antenna."
    subj = "Microwave Engineering and Antennas"
    res = predict(test_q, subj)
    print("Test Prediction Output:")
    print(json.dumps(res, indent=2))
