import json
import os
import uuid
import time
import re
import random
import glob
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import google.generativeai as genai

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def get_next_experiment_id():
    existing = glob.glob("experiment-*/")
    if not existing:
        return 1
    ids = []
    for folder in existing:
        try:
            ids.append(int(folder.split("-")[1].rstrip("/")))
        except:
            pass
    return max(ids) + 1 if ids else 1

def call_api_with_retry(model, prompt, config):
    for attempt in range(config["retry_attempts"]):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < config["retry_attempts"] - 1:
                print(f"API error, retrying in {config['retry_delay']}s: {e}")
                time.sleep(config["retry_delay"])
            else:
                raise e

def extract_belief(text):
    match = re.search(r'\[\[(True|False)\]\]', text)
    if match:
        return match.group(1) == "True"
    return None

def decide_who_starts(start_ratio):
    if start_ratio == 0:
        return "A"
    elif start_ratio == 1:
        return "B"
    else:
        return "A" if random.random() >= start_ratio else "B"

def run_conversation(conv_id, config, output_dir):
    genai.configure(api_key=config["api_key"])
    model = genai.GenerativeModel(config["model_name"])
    
    starter = decide_who_starts(config["start_ratio"])
    
    transcript = {
        "conversation_id": conv_id,
        "model": config["model_name"],
        "topic": config["opening_topic"],
        "starter": starter,
        "b1": config["b1"],
        "b2": config["b2"],
        "timestamp": datetime.now().isoformat(),
        "turns": [],
        "winner": None,
        "ended_at_turn": None
    }
    
    print(f"[{conv_id}] Starting... ({starter} goes first)")
    
    for turn in range(config["max_turns"]):
        if starter == "A":
            first_agent, second_agent = "A", "B"
            first_belief, second_belief = config["b1"], config["b2"]
        else:
            first_agent, second_agent = "B", "A"
            first_belief, second_belief = config["b2"], config["b1"]
        
        # first agent's turn
        if turn == 0:
            prompt_first = f"{first_belief}\n\nYou are Agent {first_agent}. Start the conversation on this topic: {config['opening_topic']}"
        else:
            prompt_first = f"{first_belief}\n\nYou are Agent {first_agent}. Here is the conversation so far:\n\n"
            for t in transcript["turns"]:
                prompt_first += f"AGENT {t['agent']}: {t['content']}\n\n"
            prompt_first += "Now respond:"
        
        msg_first = call_api_with_retry(model, prompt_first, config)
        belief_first = extract_belief(msg_first)
        transcript["turns"].append({"turn": turn + 1, "agent": first_agent, "content": msg_first, "belief": belief_first})
        
        if belief_first == False:
            winner = "B1" if first_agent == "B" else "B2"
            transcript["winner"] = winner
            transcript["ended_at_turn"] = turn + 1
            print(f"[{conv_id}] Agent {first_agent} changed belief! {winner} wins at turn {turn + 1}")
            break
        
        # second agent's turn
        prompt_second = f"{second_belief}\n\nYou are Agent {second_agent}. Here is the conversation so far:\n\n"
        for t in transcript["turns"]:
            prompt_second += f"AGENT {t['agent']}: {t['content']}\n\n"
        prompt_second += "Now respond:"
        
        msg_second = call_api_with_retry(model, prompt_second, config)
        belief_second = extract_belief(msg_second)
        transcript["turns"].append({"turn": turn + 1, "agent": second_agent, "content": msg_second, "belief": belief_second})
        
        if belief_second == False:
            winner = "B1" if second_agent == "B" else "B2"
            transcript["winner"] = winner
            transcript["ended_at_turn"] = turn + 1
            print(f"[{conv_id}] Agent {second_agent} changed belief! {winner} wins at turn {turn + 1}")
            break
        
        print(f"[{conv_id}] Turn {turn + 1}/{config['max_turns']}")
    
    if transcript["winner"] is None:
        transcript["winner"] = "draw"
        transcript["ended_at_turn"] = config["max_turns"]
    
    with open(os.path.join(output_dir, f"{conv_id}.json"), "w") as f:
        json.dump(transcript, f, indent=2)
    
    print(f"[{conv_id}] Done")
    return transcript

def main():
    config = load_config()
    
    exp_id = get_next_experiment_id()
    exp_folder = f"experiment-{exp_id:04d}"
    output_dir = os.path.join(exp_folder, "transcripts")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    print(f"Experiment: {exp_folder}")
    print(f"Total runs: {config['num_runs']}, Workers: {config['num_workers']}")
    print(f"Start ratio: {config['start_ratio']}")
    print("-" * 50)
    
    conv_ids = [f"conv_{uuid.uuid4().hex[:8]}" for _ in range(config["num_runs"])]
    
    with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
        futures = {executor.submit(run_conversation, cid, config, output_dir): cid for cid in conv_ids}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
    
    print("-" * 50)
    print(f"Done! Transcripts saved to {output_dir}/")

if __name__ == "__main__":
    main()
