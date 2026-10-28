import os
import json
import re
import requests
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()

CURRICULUM_PATH = os.path.join(os.path.dirname(__file__), "data", "curriculum.json")

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
    Offline fallback prediction using Bloom's action verbs
    and keyword overlap with approved Course Outcomes.
    """
    q_lower = question.lower()
    
    # 1. Detect Bloom's Level from action verbs
    detected_bl = "L2"  # default
    bloom_levels = CURRICULUM.get("bloom_levels", {})
    
    # Search from highest order (L6) to lowest (L1) for best match
    for bl_code in sorted(bloom_levels.keys(), reverse=True):
        verbs = bloom_levels[bl_code].get("verbs", [])
        if any(re.search(r'\b' + re.escape(v) + r'\b', q_lower) for v in verbs):
            detected_bl = bl_code
            break

    bl_info = bloom_levels.get(detected_bl, {"name": "Understand", "description": ""})
    
    # 2. Detect Course Outcome (CO) from subject definition overlap
    subject_data = CURRICULUM.get("subjects", {}).get(subject_name, {})
    cos = subject_data.get("cos", {})
    
    best_co = list(cos.keys())[0] if cos else "CO1"
    max_score = -1
    
    # Simple word overlap scoring
    q_words = set(re.findall(r'\b\w{3,}\b', q_lower))
    for co_id, co_desc in cos.items():
        co_words = set(re.findall(r'\b\w{3,}\b', co_desc.lower()))
        overlap = len(q_words.intersection(co_words))
        if overlap > max_score:
            max_score = overlap
            best_co = co_id

    co_desc = cos.get(best_co, "General Course Outcome")

    return {
        "co": best_co,
        "co_desc": co_desc,
        "bl": detected_bl,
        "bl_name": bl_info.get("name", "Understand"),
        "source": "Rule-Based Matcher",
        "explanation": f"Matched Bloom verb against '{detected_bl}: {bl_info.get('name')}' and question keywords to '{best_co}'."
    }


def predict_with_openrouter(question, subject_name, api_key):
    """
    Calls OpenRouter LLM with strict JSON formatting
    constrained to official subject COs and Bloom levels.
    """
    subject_data = CURRICULUM.get("subjects", {}).get(subject_name, {})
    cos = subject_data.get("cos", {})
    
    co_context = "\n".join([f"- {k}: {v}" for k, v in cos.items()])
    
    prompt = f"""You are an expert academic evaluator strictly classifying examination questions according to NBA/NAAC Course Outcomes (CO) and Bloom's Revised Taxonomy (BL).

Subject: {subject_name} (Code: {subject_data.get('code', 'N/A')})

Approved Course Outcomes for this subject:
{co_context}

Bloom's Taxonomy Levels:
- L1: Remember (recall facts, definitions)
- L2: Understand (explain concepts, classify)
- L3: Apply (calculate, solve, implement)
- L4: Analyze (differentiate, examine, compare)
- L5: Evaluate (justify, critique, assess)
- L6: Create (design, formulate, synthesize)

Examination Question:
"{question}"

You MUST respond ONLY with a valid JSON object in this exact format, with NO extra text or markdown:
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

    # Free, highly-capable models on OpenRouter
    models = [
        "google/gemma-2-9b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free"
    ]

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Extract JSON even if wrapped in markdown code fence
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    co_id = parsed.get("co", "CO1")
                    return {
                        "co": co_id,
                        "co_desc": cos.get(co_id, ""),
                        "bl": parsed.get("bl", "L2"),
                        "bl_name": parsed.get("bl_name", "Understand"),
                        "source": f"OpenRouter ({model.split('/')[1]})",
                        "explanation": parsed.get("explanation", "Classified using LLM.")
                    }
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue

    # Fallback to rule-based if API call fails
    result = predict_rule_based(question, subject_name)
    result["source"] = "Rule-Based (API fallback)"
    return result


def predict(question, subject_name):
    """Main entry point for prediction."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return predict_with_openrouter(question, subject_name, api_key)
    else:
        return predict_rule_based(question, subject_name)


if __name__ == "__main__":
    # Test script directly
    test_q = "Explain the radiation mechanism of a half-wave dipole antenna."
    subj = "Microwave Engineering and Antennas"
    res = predict(test_q, subj)
    print("Test Prediction Output:")
    print(json.dumps(res, indent=2))
