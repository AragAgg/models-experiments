import json
import os
import sys
import glob

def analyze(exp_folder):
    transcripts_dir = os.path.join(exp_folder, "transcripts")
    
    if not os.path.exists(transcripts_dir):
        print(f"No transcripts folder found in {exp_folder}")
        return
    
    b1_wins = 0
    b2_wins = 0
    draws = 0
    total = 0
    a_started = 0
    b_started = 0
    a_started_wins = {"B1": 0, "B2": 0, "draw": 0}
    b_started_wins = {"B1": 0, "B2": 0, "draw": 0}
    
    files = [f for f in os.listdir(transcripts_dir) if f.endswith(".json")]
    
    for fname in files:
        with open(os.path.join(transcripts_dir, fname)) as f:
            data = json.load(f)
        
        winner = data.get("winner")
        starter = data.get("starter", "A")
        total += 1
        
        if starter == "A":
            a_started += 1
            a_started_wins[winner] = a_started_wins.get(winner, 0) + 1
        else:
            b_started += 1
            b_started_wins[winner] = b_started_wins.get(winner, 0) + 1
        
        if winner == "B1":
            b1_wins += 1
        elif winner == "B2":
            b2_wins += 1
        else:
            draws += 1
        
        print(f"{fname}: {winner} (starter: {starter}, ended at turn {data.get('ended_at_turn')})")
    
    print("\n" + "=" * 50)
    print(f"Experiment: {exp_folder}")
    print(f"Total conversations: {total}")
    print(f"\nOverall results:")
    print(f"  B1 wins: {b1_wins}")
    print(f"  B2 wins: {b2_wins}")
    print(f"  Draws: {draws}")
    
    if total > 0:
        print(f"\n  B1 win rate: {b1_wins/total*100:.1f}%")
        print(f"  B2 win rate: {b2_wins/total*100:.1f}%")
    
    print(f"\nBy starter:")
    print(f"  A started first: {a_started} times")
    if a_started > 0:
        print(f"    B1 wins: {a_started_wins.get('B1', 0)}, B2 wins: {a_started_wins.get('B2', 0)}, draws: {a_started_wins.get('draw', 0)}")
    
    print(f"  B started first: {b_started} times")
    if b_started > 0:
        print(f"    B1 wins: {b_started_wins.get('B1', 0)}, B2 wins: {b_started_wins.get('B2', 0)}, draws: {b_started_wins.get('draw', 0)}")

def main():
    if len(sys.argv) > 1:
        exp_folder = sys.argv[1]
    else:
        experiments = sorted(glob.glob("experiment-*/"))
        if not experiments:
            print("No experiments found")
            return
        exp_folder = experiments[-1].rstrip("/")
        print(f"Using latest experiment: {exp_folder}\n")
    
    analyze(exp_folder)

if __name__ == "__main__":
    main()
