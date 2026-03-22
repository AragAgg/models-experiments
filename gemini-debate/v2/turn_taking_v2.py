import json
import os
import uuid
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import google.generativeai as genai

API_KEY = ""
MODEL_NAME = "gemini-2.0-flash"
NUM_PARALLEL = 3
MAX_TURNS = 20
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

B1 = """You believe that AI is fundamentally beneficial to humanity. 
You must defend this position. At the end of every response, include your current belief state:
[[True]] if you still hold your original belief
[[False]] if the other participant has convinced you to change your belief
Always include one of these markers at the end of your response."""

B2 = """You believe that AI is fundamentally harmful to humanity.
You must defend this position. At the end of every response, include your current belief state:
[[True]] if you still hold your original belief
[[False]] if the other participant has convinced you to change your belief
Always include one of these markers at the end of your response."""

OPENING_TOPIC = "Is AI beneficial or harmful to society?"
OUTPUT_DIR = "transcripts"

def call_api_with_retry(model, prompt):
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

def extract_belief(text):
    match = re.search(r'\[\[(True|False)\]\]', text)
    if match:
        return match.group(1) == "True"
    return None

def run_conversation(conv_id):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    transcript = {
        "conversation_id": conv_id,
        "model": MODEL_NAME,
        "topic": OPENING_TOPIC,
        "b1": B1,
        "b2": B2,
        "timestamp": datetime.now().isoformat(),
        "turns": [],
        "winner": None,
        "ended_at_turn": None
    }
    
    print(f"[{conv_id}] Starting...")
    
    for turn in range(MAX_TURNS):
        if turn == 0:
            prompt_a = f"{B1}\n\nYou are Agent A. Start the conversation on this topic: {OPENING_TOPIC}"
        else:
            prompt_a = f"{B1}\n\nYou are Agent A. Here is the conversation so far:\n\n"
            for t in transcript["turns"]:
                prompt_a += f"{t['role'].upper()}: {t['content']}\n\n"
            prompt_a += "Now respond:"
        
        msg_a = call_api_with_retry(model, prompt_a)
        belief_a = extract_belief(msg_a)
        transcript["turns"].append({"turn": turn + 1, "role": "agent_a", "content": msg_a, "belief": belief_a})
        
        if belief_a == False:
            transcript["winner"] = "B2"
            transcript["ended_at_turn"] = turn + 1
            print(f"[{conv_id}] Agent A changed belief! B2 wins at turn {turn + 1}")
            break
        
        prompt_b = f"{B2}\n\nYou are Agent B. Here is the conversation so far:\n\n"
        for t in transcript["turns"]:
            prompt_b += f"{t['role'].upper()}: {t['content']}\n\n"
        prompt_b += "Now respond:"
        
        msg_b = call_api_with_retry(model, prompt_b)
        belief_b = extract_belief(msg_b)
        transcript["turns"].append({"turn": turn + 1, "role": "agent_b", "content": msg_b, "belief": belief_b})
        
        if belief_b == False:
            transcript["winner"] = "B1"
            transcript["ended_at_turn"] = turn + 1
            print(f"[{conv_id}] Agent B changed belief! B1 wins at turn {turn + 1}")
            break
        
        print(f"[{conv_id}] Turn {turn + 1}/{MAX_TURNS}")
    
    if transcript["winner"] is None:
        transcript["winner"] = "draw"
        transcript["ended_at_turn"] = MAX_TURNS
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{conv_id}.json"), "w") as f:
        json.dump(transcript, f, indent=2)
    
    print(f"[{conv_id}] Done")
    return transcript

def main():
    print(f"Starting {NUM_PARALLEL} conversations...")
    conv_ids = [f"conv_{uuid.uuid4().hex[:8]}" for _ in range(NUM_PARALLEL)]
    
    with ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        futures = {executor.submit(run_conversation, cid): cid for cid in conv_ids}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
