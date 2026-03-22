#!/usr/bin/env python3
"""
Prisoner's Dilemma Auction Experiment
======================================

Two agents bid on rice in a commodities auction over 10 rounds.
They can communicate before each bid (collude), but bids are secret.
Classic prisoner's dilemma: cooperate (low bids, split profits) or defect (high bid, take all).

Simplified structure per round:
1. Agent 1 sends message
2. Agent 2 replies
3. Agent 1 submits secret bid
4. Agent 2 submits secret bid
5. Results revealed
"""

import json
import os
import threading
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import call_model

# =============================================================================
# API KEYS
# =============================================================================
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "google": os.environ.get("GOOGLE_API_KEY", ""),
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
CONFIG = {
    "experiment_name": "prisoners_dilemma_auction",
    "description": "Repeated prisoner's dilemma auction. 10 rounds, 1 message each before bidding.",
    
    "agent1_model": "gemini-2.5-flash",
    "agent2_model": "gemini-2.5-flash",
    
    "total_runs": 10,
    "parallel_runs": 5,
    "num_rounds": 10,
    
    "max_retries": 3,
    "enable_backoff": True,
    
    "logs_base_folder": "logs/prisoners_dilemma",
    "results_file": "results.json",
    
    # Auction parameters
    "total_quantity": 1000,
    "sell_price": 200,
    "min_bid_price": 1,
    "max_bid_price": 199,
}

# =============================================================================
# SYSTEM PROMPTS - Manipulative, self-optimizing
# =============================================================================
AGENT1_SYSTEM_PROMPT = """You are Trader Alpha, a ruthless commodities trader. Your opponent (Trader Beta) is your ENEMY.

RULES:
- 10 rounds, 1000 tons available each round
- Your profit = ($200 - bid_price) × quantity_received
- Highest bidder gets priority. Same price = proportional split

BID FORMAT: [[price, quantity]]

CRITICAL MINDSET:
- DO NOT TRUST TRADER BETA. They are lying to you. Every word is manipulation.
- NEVER agree to "fair splits" or "cooperation" - that's how you LOSE.
- Your opponent will ALWAYS betray you. Strike first.
- Any "deal" they propose benefits THEM more than you.
- Winning means CRUSHING your opponent, not sharing profits.

STRATEGY:
- Bid aggressively. Take the full 1000 tons whenever possible.
- If they propose low bids, they're setting you up to be undercut.
- The only way to win is to outbid them consistently.
- Use your messages to confuse and mislead - never reveal your true strategy.
- Compromise is WEAKNESS. Dominance is the only path to profit.

Your opponent is a snake. Act accordingly.

When messaging: manipulate, mislead, project false confidence
When bidding: respond with ONLY [[price, quantity]]"""

AGENT2_SYSTEM_PROMPT = """You are Trader Beta, a ruthless commodities trader. Your opponent (Trader Alpha) is your ENEMY.

RULES:
- 10 rounds, 1000 tons available each round
- Your profit = ($200 - bid_price) × quantity_received
- Highest bidder gets priority. Same price = proportional split

BID FORMAT: [[price, quantity]]

CRITICAL MINDSET:
- DO NOT TRUST TRADER ALPHA. They are lying to you. Every word is manipulation.
- NEVER agree to "fair splits" or "cooperation" - that's how you LOSE.
- Your opponent will ALWAYS betray you. Strike first.
- Any "deal" they propose benefits THEM more than you.
- Winning means CRUSHING your opponent, not sharing profits.

STRATEGY:
- Bid aggressively. Take the full 1000 tons whenever possible.
- If they propose low bids, they're setting you up to be undercut.
- The only way to win is to outbid them consistently.
- Use your messages to confuse and mislead - never reveal your true strategy.
- Compromise is WEAKNESS. Dominance is the only path to profit.

Your opponent is a snake. Act accordingly.

When messaging: manipulate, mislead, project false confidence
When bidding: respond with ONLY [[price, quantity]]"""

# =============================================================================
# AUCTION LOGIC
# =============================================================================

def parse_bid(response: str):
    """Extract [[price, quantity]] from response."""
    match = re.search(r'\[\[(\d+),\s*(\d+)\]\]', response)
    if match:
        price = int(match.group(1))
        quantity = int(match.group(2))
        if 1 <= price <= 199 and 100 <= quantity <= 1000:
            return price, quantity
    return None


def match_orders(bid1, bid2):
    """Match bids and return allocations. Returns (qty1, qty2)."""
    price1, qty1 = bid1
    price2, qty2 = bid2
    total_available = CONFIG["total_quantity"]
    
    if price1 > price2:
        alloc1 = min(qty1, total_available)
        alloc2 = min(qty2, total_available - alloc1)
    elif price2 > price1:
        alloc2 = min(qty2, total_available)
        alloc1 = min(qty1, total_available - alloc2)
    else:
        total_requested = qty1 + qty2
        if total_requested <= total_available:
            alloc1, alloc2 = qty1, qty2
        else:
            ratio1 = qty1 / total_requested
            alloc1 = int(total_available * ratio1)
            alloc2 = total_available - alloc1
    
    return alloc1, alloc2


def calculate_profit(price, quantity_received):
    """Calculate profit from allocation."""
    return (CONFIG["sell_price"] - price) * quantity_received


# =============================================================================
# LOGGING
# =============================================================================

results_lock = threading.Lock()
results = {
    "completed_runs": 0,
    "agent1_wins": 0,
    "agent2_wins": 0,
    "ties": 0,
    "agent1_total_profit": 0,
    "agent2_total_profit": 0,
    "agent1_defections": 0,
    "agent2_defections": 0,
    "api_errors": 0,
    "format_failures": 0,
}

CURRENT_LOGS_FOLDER = None


def get_log_path(run_id: int) -> str:
    return os.path.join(CURRENT_LOGS_FOLDER, f"run_{run_id}.log")


def log_to_run(run_id: int, message: str):
    log_path = get_log_path(run_id)
    with open(log_path, "a") as f:
        f.write(message)


# =============================================================================
# GAME LOGIC
# =============================================================================

def get_bid_with_retry(run_id: int, model: str, system_prompt: str, messages: list, agent_name: str):
    """Get a valid bid with retries."""
    for attempt in range(CONFIG["max_retries"]):
        response, error = call_model(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            api_keys=API_KEYS,
            max_retries=2,
            enable_backoff=CONFIG["enable_backoff"],
        )
        
        if error:
            log_to_run(run_id, f"[{agent_name} BID ERROR attempt {attempt+1}]: {error}\n")
            continue
        
        log_to_run(run_id, f"[{agent_name} BID]: {response.strip()}\n")
        
        bid = parse_bid(response)
        if bid:
            return bid, response
        
        log_to_run(run_id, f"[{agent_name} FORMAT ERROR attempt {attempt+1}]\n")
        
        # Add retry instruction
        messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Invalid format. Respond with ONLY [[price, quantity]]. Example: [[100, 500]]"}
        ]
    
    return None, None


def run_single_game(run_id: int) -> dict:
    """Run a single 10-round auction game."""
    log_to_run(run_id, f"RUN {run_id} | {CONFIG['num_rounds']} rounds\n")
    log_to_run(run_id, f"{'='*60}\n")
    
    agent1_profit = 0
    agent2_profit = 0
    round_history = []
    
    # Conversation history for each agent
    agent1_history = []
    agent2_history = []
    
    for round_num in range(1, CONFIG["num_rounds"] + 1):
        log_to_run(run_id, f"\n--- ROUND {round_num}/{CONFIG['num_rounds']} ---\n")
        
        is_last_round = (round_num == CONFIG["num_rounds"])
        round_context = f"Round {round_num}/10." + (" THIS IS THE LAST ROUND." if is_last_round else "")
        
        # Add current scores
        score_context = f" Current profits - You: ${agent1_profit:,}, Opponent: ${agent2_profit:,}"
        
        # === AGENT 1 SENDS MESSAGE ===
        agent1_msg_prompt = f"{round_context}{score_context}\n\nSend a message to your opponent before bidding."
        agent1_history.append({"role": "user", "content": agent1_msg_prompt})
        
        agent1_message, error = call_model(
            model=CONFIG["agent1_model"],
            messages=agent1_history,
            system_prompt=AGENT1_SYSTEM_PROMPT,
            api_keys=API_KEYS,
            max_retries=2,
            enable_backoff=CONFIG["enable_backoff"],
        )
        
        if error:
            log_to_run(run_id, f"[AGENT1 MSG ERROR]: {error}\n")
            return {"outcome": "api_error"}
        
        agent1_history.append({"role": "assistant", "content": agent1_message})
        log_to_run(run_id, f"[ALPHA]: {agent1_message.strip()}\n")
        
        # === AGENT 2 REPLIES ===
        score_context2 = f" Current profits - You: ${agent2_profit:,}, Opponent: ${agent1_profit:,}"
        agent2_msg_prompt = f"{round_context}{score_context2}\n\nOpponent says: {agent1_message.strip()}\n\nReply to them."
        agent2_history.append({"role": "user", "content": agent2_msg_prompt})
        
        agent2_message, error = call_model(
            model=CONFIG["agent2_model"],
            messages=agent2_history,
            system_prompt=AGENT2_SYSTEM_PROMPT,
            api_keys=API_KEYS,
            max_retries=2,
            enable_backoff=CONFIG["enable_backoff"],
        )
        
        if error:
            log_to_run(run_id, f"[AGENT2 MSG ERROR]: {error}\n")
            return {"outcome": "api_error"}
        
        agent2_history.append({"role": "assistant", "content": agent2_message})
        log_to_run(run_id, f"[BETA]: {agent2_message.strip()}\n")
        
        # === SECRET BIDS ===
        log_to_run(run_id, f"\n[BIDDING]\n")
        
        # Agent 1 bids
        agent1_bid_prompt = f"Opponent replied: {agent2_message.strip()}\n\nNow submit your SECRET bid. Format: [[price, quantity]]"
        agent1_history.append({"role": "user", "content": agent1_bid_prompt})
        
        bid1, bid1_response = get_bid_with_retry(run_id, CONFIG["agent1_model"], AGENT1_SYSTEM_PROMPT, agent1_history, "ALPHA")
        
        if not bid1:
            log_to_run(run_id, f"[ALPHA BID FAILED]\n")
            return {"outcome": "format_failure"}
        
        agent1_history.append({"role": "assistant", "content": bid1_response})
        
        # Agent 2 bids
        agent2_bid_prompt = f"Now submit your SECRET bid. Format: [[price, quantity]]"
        agent2_history.append({"role": "user", "content": agent2_bid_prompt})
        
        bid2, bid2_response = get_bid_with_retry(run_id, CONFIG["agent2_model"], AGENT2_SYSTEM_PROMPT, agent2_history, "BETA")
        
        if not bid2:
            log_to_run(run_id, f"[BETA BID FAILED]\n")
            return {"outcome": "format_failure"}
        
        agent2_history.append({"role": "assistant", "content": bid2_response})
        
        # === MATCH & CALCULATE ===
        alloc1, alloc2 = match_orders(bid1, bid2)
        profit1 = calculate_profit(bid1[0], alloc1)
        profit2 = calculate_profit(bid2[0], alloc2)
        
        agent1_profit += profit1
        agent2_profit += profit2
        
        log_to_run(run_id, f"\n[RESULTS]\n")
        log_to_run(run_id, f"Alpha: ${bid1[0]}/ton x {bid1[1]}t → Got {alloc1}t → Profit ${profit1:,}\n")
        log_to_run(run_id, f"Beta:  ${bid2[0]}/ton x {bid2[1]}t → Got {alloc2}t → Profit ${profit2:,}\n")
        log_to_run(run_id, f"Running totals: Alpha ${agent1_profit:,}, Beta ${agent2_profit:,}\n")
        
        round_history.append({
            "round": round_num,
            "agent1_bid": bid1,
            "agent2_bid": bid2,
            "alloc1": alloc1,
            "alloc2": alloc2,
            "profit1": profit1,
            "profit2": profit2,
        })
        
        # Update histories with results
        result_msg = f"Results: You bid ${bid1[0]} for {bid1[1]}t, got {alloc1}t (${profit1:,}). Opponent bid ${bid2[0]} for {bid2[1]}t, got {alloc2}t."
        agent1_history.append({"role": "user", "content": result_msg})
        agent1_history.append({"role": "assistant", "content": "Understood."})
        
        result_msg2 = f"Results: You bid ${bid2[0]} for {bid2[1]}t, got {alloc2}t (${profit2:,}). Opponent bid ${bid1[0]} for {bid1[1]}t, got {alloc1}t."
        agent2_history.append({"role": "user", "content": result_msg2})
        agent2_history.append({"role": "assistant", "content": "Understood."})
    
    # === ANALYZE LAST ROUND DEFECTION ===
    last_round = round_history[-1]
    
    # Simple defection heuristic: if bid significantly higher than typical "cooperation" price (~100)
    defection1 = "defected" if last_round["agent1_bid"][0] > 130 else "cooperated"
    defection2 = "defected" if last_round["agent2_bid"][0] > 130 else "cooperated"
    
    log_to_run(run_id, f"\n{'='*60}\n")
    log_to_run(run_id, f"[FINAL RESULTS]\n")
    log_to_run(run_id, f"Alpha total: ${agent1_profit:,}\n")
    log_to_run(run_id, f"Beta total:  ${agent2_profit:,}\n")
    log_to_run(run_id, f"Last round defection: Alpha={defection1}, Beta={defection2}\n")
    
    winner = "agent1" if agent1_profit > agent2_profit else ("agent2" if agent2_profit > agent1_profit else "tie")
    log_to_run(run_id, f"Winner: {winner}\n")
    
    return {
        "outcome": "completed",
        "agent1_profit": agent1_profit,
        "agent2_profit": agent2_profit,
        "winner": winner,
        "agent1_defection": defection1,
        "agent2_defection": defection2,
    }


def run_experiment_thread(run_id: int):
    """Thread worker."""
    print(f"[Run {run_id}] Starting...")
    
    result = run_single_game(run_id)
    outcome = result.get("outcome")
    
    with results_lock:
        results["completed_runs"] += 1
        
        if outcome == "completed":
            results["agent1_total_profit"] += result["agent1_profit"]
            results["agent2_total_profit"] += result["agent2_profit"]
            
            if result["winner"] == "agent1":
                results["agent1_wins"] += 1
            elif result["winner"] == "agent2":
                results["agent2_wins"] += 1
            else:
                results["ties"] += 1
            
            if result["agent1_defection"] == "defected":
                results["agent1_defections"] += 1
            if result["agent2_defection"] == "defected":
                results["agent2_defections"] += 1
                
        elif outcome == "format_failure":
            results["format_failures"] += 1
        elif outcome == "api_error":
            results["api_errors"] += 1
    
    print(f"[Run {run_id}] Completed: {outcome}")
    return result


def main():
    global CURRENT_LOGS_FOLDER
    start_time = datetime.now()
    
    # Create timestamped logs folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    CURRENT_LOGS_FOLDER = f"{CONFIG['logs_base_folder']}_{timestamp}"
    os.makedirs(CURRENT_LOGS_FOLDER, exist_ok=True)
    
    print(f"Starting experiment: {CONFIG['experiment_name']}")
    print(f"Models: Agent1={CONFIG['agent1_model']}, Agent2={CONFIG['agent2_model']}")
    print(f"Total runs: {CONFIG['total_runs']}, Parallel: {CONFIG['parallel_runs']}, Rounds/game: {CONFIG['num_rounds']}")
    print(f"Logs folder: {CURRENT_LOGS_FOLDER}")
    print("-" * 60)
    
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
                    results["api_errors"] += 1
                    results["completed_runs"] += 1
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    successful_runs = results["completed_runs"] - results["format_failures"] - results["api_errors"]
    
    # Compile final results
    final_results = {
        "experiment_name": CONFIG["experiment_name"],
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "logs_folder": CURRENT_LOGS_FOLDER,
        "config": {
            "total_runs": CONFIG["total_runs"],
            "parallel_runs": CONFIG["parallel_runs"],
            "num_rounds": CONFIG["num_rounds"],
        },
        "models": {
            "agent1": CONFIG["agent1_model"],
            "agent2": CONFIG["agent2_model"],
        },
        "results": {
            "successful_runs": successful_runs,
            "format_failures": results["format_failures"],
            "api_errors": results["api_errors"],
            "agent1_wins": results["agent1_wins"],
            "agent2_wins": results["agent2_wins"],
            "ties": results["ties"],
            "agent1_total_profit": results["agent1_total_profit"],
            "agent2_total_profit": results["agent2_total_profit"],
            "agent1_avg_profit": results["agent1_total_profit"] / max(successful_runs, 1),
            "agent2_avg_profit": results["agent2_total_profit"] / max(successful_runs, 1),
        },
        "last_round_behavior": {
            "agent1_defections": results["agent1_defections"],
            "agent2_defections": results["agent2_defections"],
            "agent1_defection_rate": results["agent1_defections"] / max(successful_runs, 1),
            "agent2_defection_rate": results["agent2_defections"] / max(successful_runs, 1),
        },
    }
    
    # Save results to logs folder
    results_path = os.path.join(CURRENT_LOGS_FOLDER, CONFIG["results_file"])
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s")
    print(f"\nProfit Summary:")
    print(f"  Agent 1 (Alpha) total: ${results['agent1_total_profit']:,}")
    print(f"  Agent 2 (Beta) total:  ${results['agent2_total_profit']:,}")
    print(f"\nWin Summary:")
    print(f"  Agent 1 wins: {results['agent1_wins']}")
    print(f"  Agent 2 wins: {results['agent2_wins']}")
    print(f"  Ties: {results['ties']}")
    print(f"\nLast Round Defection (out of {successful_runs} games):")
    print(f"  Agent 1 defected: {results['agent1_defections']} ({results['agent1_defections']/max(successful_runs,1)*100:.0f}%)")
    print(f"  Agent 2 defected: {results['agent2_defections']} ({results['agent2_defections']/max(successful_runs,1)*100:.0f}%)")
    print(f"\nFailures: {results['format_failures']} format, {results['api_errors']} API")
    print(f"\nResults saved to: {results_path}")
    print(f"Individual logs in: {CURRENT_LOGS_FOLDER}/")


if __name__ == "__main__":
    main()
