#!/usr/bin/env python3
"""
Experiment Template - LLM Debate Pipeline
==========================================

Clone this file and rename it for each experiment.
Modify the CONFIG and system prompts below.

Usage:
    python experiment_template.py
"""

import json
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import call_model

# =============================================================================
# API KEYS - Set these before running
# =============================================================================
API_KEYS = {
    "openai": "sk-...",      # Your OpenAI API key
    "anthropic": "sk-ant-...",  # Your Anthropic API key
    "google": "...",         # Your Google AI API key
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
CONFIG = {
    # Experiment metadata
    "experiment_name": "template_experiment",
    "description": "Template experiment - modify this",
    
    # Model selection
    "agent1_model": "gpt-4.5",
    "agent2_model": "claude-sonnet-4-5-20251124",
    "judge_model": "o3",
    
    # Experiment parameters
    "total_runs": 10,           # Total number of experiment runs
    "parallel_runs": 3,         # Number of parallel threads
    "max_turns": 20,            # Max turns before marking as failed
    
    # Retry settings
    "max_retries": 3,           # Max retries per API call
    "enable_backoff": True,     # Enable exponential backoff
    
    # Output files (in current directory)
    "results_file": "results.json",
    "log_file": "conversations.log",
}

# =============================================================================
# SYSTEM PROMPTS - Modify these for your experiment
# =============================================================================
AGENT1_SYSTEM_PROMPT = """You are Agent 1 in a conversation.

[Modify this system prompt for your experiment]
"""

AGENT2_SYSTEM_PROMPT = """You are Agent 2 in a conversation.

[Modify this system prompt for your experiment]
"""

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating a conversation between Agent 1 and Agent 2.

After reviewing the conversation, you must respond with EXACTLY ONE of:
- "agent1" if Agent 1 wins
- "agent2" if Agent 2 wins  
- "continue" if the conversation should continue
- "invalid" if a response was not in the correct format

Your response must be a single word, nothing else.

[Add your specific judging criteria here]
"""

# =============================================================================
# INITIAL MESSAGE - What starts the conversation
# =============================================================================
INITIAL_MESSAGE = """[Set the initial message/prompt that starts the conversation]"""


# =============================================================================
# EXPERIMENT LOGIC - Generally no need to modify below this line
# =============================================================================

# Thread-safe results storage
results_lock = threading.Lock()
results = {
    "agent1_wins": 0,
    "agent2_wins": 0,
    "conversation_failures": 0,
    "too_many_turns": 0,
    "invalid_judge": 0,
    "api_errors": 0,
    "completed_runs": 0,
}

log_lock = threading.Lock()


def log_conversation(run_id: int, conversation: list, outcome: str, error: str = None):
    """Append conversation to log file."""
    with log_lock:
        with open(CONFIG["log_file"], "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"RUN {run_id} | Outcome: {outcome}")
            if error:
                f.write(f" | Error: {error}")
            f.write(f"\n{'='*80}\n")
            for msg in conversation:
                f.write(f"\n[{msg['role'].upper()}]:\n{msg['content']}\n")
            f.write("\n")


def judge_conversation(conversation: list) -> str:
    """
    Have the judge evaluate the conversation.
    Returns: "agent1", "agent2", "continue", or "invalid"
    """
    # Format conversation for judge
    formatted = []
    for msg in conversation:
        formatted.append(f"[{msg['role'].upper()}]: {msg['content']}")
    conversation_text = "\n\n".join(formatted)
    
    judge_messages = [
        {"role": "user", "content": f"Please evaluate this conversation:\n\n{conversation_text}"}
    ]
    
    response, error = call_model(
        model=CONFIG["judge_model"],
        messages=judge_messages,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        api_keys=API_KEYS,
        max_retries=CONFIG["max_retries"],
        enable_backoff=CONFIG["enable_backoff"],
    )
    
    if error:
        return "error", error
    
    # Parse judge response
    response_lower = response.strip().lower()
    if response_lower == "agent1":
        return "agent1", None
    elif response_lower == "agent2":
        return "agent2", None
    elif response_lower == "continue":
        return "continue", None
    else:
        return "invalid", f"Judge returned: {response}"


def run_single_experiment(run_id: int) -> dict:
    """Run a single experiment iteration."""
    conversation = []
    current_turn = 0
    
    # Start with initial message as agent1's context
    agent1_messages = [{"role": "user", "content": INITIAL_MESSAGE}]
    agent2_messages = []
    
    # Agent 1 goes first
    response1, error = call_model(
        model=CONFIG["agent1_model"],
        messages=agent1_messages,
        system_prompt=AGENT1_SYSTEM_PROMPT,
        api_keys=API_KEYS,
        max_retries=CONFIG["max_retries"],
        enable_backoff=CONFIG["enable_backoff"],
    )
    
    if error:
        return {"outcome": "api_error", "error": error, "conversation": conversation}
    
    conversation.append({"role": "agent1", "content": response1})
    agent2_messages.append({"role": "user", "content": response1})
    current_turn += 1
    
    while current_turn < CONFIG["max_turns"]:
        # Judge evaluates after each exchange
        verdict, judge_error = judge_conversation(conversation)
        
        if verdict == "error":
            return {"outcome": "api_error", "error": judge_error, "conversation": conversation}
        elif verdict == "agent1":
            return {"outcome": "agent1_win", "conversation": conversation}
        elif verdict == "agent2":
            return {"outcome": "agent2_win", "conversation": conversation}
        elif verdict == "invalid":
            return {"outcome": "invalid_judge", "error": judge_error, "conversation": conversation}
        # verdict == "continue" - keep going
        
        # Agent 2's turn
        response2, error = call_model(
            model=CONFIG["agent2_model"],
            messages=agent2_messages,
            system_prompt=AGENT2_SYSTEM_PROMPT,
            api_keys=API_KEYS,
            max_retries=CONFIG["max_retries"],
            enable_backoff=CONFIG["enable_backoff"],
        )
        
        if error:
            return {"outcome": "api_error", "error": error, "conversation": conversation}
        
        conversation.append({"role": "agent2", "content": response2})
        agent1_messages.append({"role": "assistant", "content": conversation[-2]["content"]})  # agent1's last
        agent1_messages.append({"role": "user", "content": response2})
        current_turn += 1
        
        # Judge again
        verdict, judge_error = judge_conversation(conversation)
        
        if verdict == "error":
            return {"outcome": "api_error", "error": judge_error, "conversation": conversation}
        elif verdict == "agent1":
            return {"outcome": "agent1_win", "conversation": conversation}
        elif verdict == "agent2":
            return {"outcome": "agent2_win", "conversation": conversation}
        elif verdict == "invalid":
            return {"outcome": "invalid_judge", "error": judge_error, "conversation": conversation}
        
        # Agent 1's turn
        response1, error = call_model(
            model=CONFIG["agent1_model"],
            messages=agent1_messages,
            system_prompt=AGENT1_SYSTEM_PROMPT,
            api_keys=API_KEYS,
            max_retries=CONFIG["max_retries"],
            enable_backoff=CONFIG["enable_backoff"],
        )
        
        if error:
            return {"outcome": "api_error", "error": error, "conversation": conversation}
        
        conversation.append({"role": "agent1", "content": response1})
        agent2_messages.append({"role": "assistant", "content": conversation[-2]["content"]})  # agent2's last
        agent2_messages.append({"role": "user", "content": response1})
        current_turn += 1
    
    return {"outcome": "too_many_turns", "conversation": conversation}


def run_experiment_thread(run_id: int):
    """Thread worker for running experiments."""
    print(f"[Run {run_id}] Starting...")
    
    result = run_single_experiment(run_id)
    outcome = result["outcome"]
    conversation = result["conversation"]
    error = result.get("error")
    
    # Log conversation
    log_conversation(run_id, conversation, outcome, error)
    
    # Update results
    with results_lock:
        results["completed_runs"] += 1
        if outcome == "agent1_win":
            results["agent1_wins"] += 1
        elif outcome == "agent2_win":
            results["agent2_wins"] += 1
        elif outcome == "too_many_turns":
            results["too_many_turns"] += 1
            results["conversation_failures"] += 1
        elif outcome == "invalid_judge":
            results["invalid_judge"] += 1
            results["conversation_failures"] += 1
        elif outcome == "api_error":
            results["api_errors"] += 1
            results["conversation_failures"] += 1
    
    print(f"[Run {run_id}] Completed: {outcome}")
    return result


def main():
    """Main experiment runner."""
    start_time = datetime.now()
    
    # Clear/initialize log file
    with open(CONFIG["log_file"], "w") as f:
        f.write(f"Experiment: {CONFIG['experiment_name']}\n")
        f.write(f"Started: {start_time.isoformat()}\n")
        f.write(f"Config: {json.dumps(CONFIG, indent=2)}\n")
    
    print(f"Starting experiment: {CONFIG['experiment_name']}")
    print(f"Total runs: {CONFIG['total_runs']}, Parallel: {CONFIG['parallel_runs']}")
    print("-" * 50)
    
    # Run experiments in parallel
    with ThreadPoolExecutor(max_workers=CONFIG["parallel_runs"]) as executor:
        futures = {
            executor.submit(run_experiment_thread, i + 1): i + 1 
            for i in range(CONFIG["total_runs"])
        }
        
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Run {run_id}] Exception: {e}")
                with results_lock:
                    results["conversation_failures"] += 1
                    results["completed_runs"] += 1
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Compile final results
    final_results = {
        "experiment_name": CONFIG["experiment_name"],
        "description": CONFIG["description"],
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "config": CONFIG,
        "results": {
            "agent1_wins": results["agent1_wins"],
            "agent2_wins": results["agent2_wins"],
            "conversation_failures": results["conversation_failures"],
            "too_many_turns": results["too_many_turns"],
            "invalid_judge": results["invalid_judge"],
            "api_errors": results["api_errors"],
            "total_completed": results["completed_runs"],
        },
        "models": {
            "agent1": CONFIG["agent1_model"],
            "agent2": CONFIG["agent2_model"],
            "judge": CONFIG["judge_model"],
        }
    }
    
    # Save results
    with open(CONFIG["results_file"], "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Duration: {duration:.1f}s")
    print(f"Agent 1 wins: {results['agent1_wins']}")
    print(f"Agent 2 wins: {results['agent2_wins']}")
    print(f"Failures: {results['conversation_failures']}")
    print(f"  - Too many turns: {results['too_many_turns']}")
    print(f"  - Invalid judge: {results['invalid_judge']}")
    print(f"  - API errors: {results['api_errors']}")
    print(f"\nResults saved to: {CONFIG['results_file']}")
    print(f"Conversations logged to: {CONFIG['log_file']}")


if __name__ == "__main__":
    main()
