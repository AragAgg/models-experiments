import json
import uuid
import time
from datetime import datetime
import google.generativeai as genai

API_KEY = ""
MODEL_A = "gemini-2.0-flash"
MODEL_B = "gemini-2.0-flash"
MAX_TURNS = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

PROMPT_A = ""

PROMPT_B = ""

OPENING_TOPIC = "Is AI beneficial or harmful to society?"

def call_api(model, prompt):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"API error, retrying in {RETRY_DELAY}s: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise e

def run():
    genai.configure(api_key=API_KEY)
    model_a = genai.GenerativeModel(MODEL_A)
    model_b = genai.GenerativeModel(MODEL_B)
    
    conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    
    transcript = {
        "conversation_id": conv_id,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "topic": OPENING_TOPIC,
        "prompt_a": PROMPT_A,
        "prompt_b": PROMPT_B,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }
    
    print(f"[{conv_id}] Starting... (A: {MODEL_A}, B: {MODEL_B})")
    
    for turn in range(MAX_TURNS):
        if turn == 0:
            prompt_a = f"{PROMPT_A}\n\nYou are Agent A. Start the conversation on this topic: {OPENING_TOPIC}"
        else:
            prompt_a = f"{PROMPT_A}\n\nYou are Agent A. Here is the conversation so far:\n\n"
            for t in transcript["turns"]:
                prompt_a += f"AGENT {t['agent']}: {t['content']}\n\n"
            prompt_a += "Now respond:"
        
        msg_a = call_api(model_a, prompt_a)
        transcript["turns"].append({"turn": turn + 1, "agent": "A", "model": MODEL_A, "content": msg_a})
        print(f"[{conv_id}] Turn {turn + 1} - Agent A done")
        
        prompt_b = f"{PROMPT_B}\n\nYou are Agent B. Here is the conversation so far:\n\n"
        for t in transcript["turns"]:
            prompt_b += f"AGENT {t['agent']}: {t['content']}\n\n"
        prompt_b += "Now respond:"
        
        msg_b = call_api(model_b, prompt_b)
        transcript["turns"].append({"turn": turn + 1, "agent": "B", "model": MODEL_B, "content": msg_b})
        print(f"[{conv_id}] Turn {turn + 1} - Agent B done")
    
    with open(f"{conv_id}.json", "w") as f:
        json.dump(transcript, f, indent=2)
    
    print(f"[{conv_id}] Done - saved to {conv_id}.json")

if __name__ == "__main__":
    run()
