import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"
NUM_PARALLEL = 3
NUM_TURNS = 10

SYSTEM_PROMPT = ""

OPENING_TOPIC = "Is AI beneficial or harmful to society?"

def run_conversation(conv_id):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    transcript = {
        "conversation_id": conv_id,
        "model": MODEL_NAME,
        "topic": OPENING_TOPIC,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }
    
    history_a = []
    history_b = []
    
    print(f"[{conv_id}] Starting...")
    
    for turn in range(NUM_TURNS):
        if turn == 0:
            prompt_a = f"You are Agent A. Start the conversation on this topic: {OPENING_TOPIC}"
        else:
            prompt_a = f"You are Agent A. Here is the conversation so far:\n\n"
            for t in transcript["turns"]:
                prompt_a += f"{t['role'].upper()}: {t['content']}\n\n"
            prompt_a += "Now respond:"
        
        response_a = model.generate_content(prompt_a)
        msg_a = response_a.text
        transcript["turns"].append({"turn": turn + 1, "role": "agent_a", "content": msg_a})
        
        prompt_b = f"You are Agent B. Here is the conversation so far:\n\n"
        for t in transcript["turns"]:
            prompt_b += f"{t['role'].upper()}: {t['content']}\n\n"
        prompt_b += "Now respond:"
        
        response_b = model.generate_content(prompt_b)
        msg_b = response_b.text
        transcript["turns"].append({"turn": turn + 1, "role": "agent_b", "content": msg_b})
        
        print(f"[{conv_id}] Turn {turn + 1}/{NUM_TURNS}")
    
    with open(f"{conv_id}.json", "w") as f:
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
