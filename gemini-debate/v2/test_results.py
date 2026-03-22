import json
import os

TRANSCRIPTS_DIR = "transcripts"

def analyze():
    b1_wins = 0
    b2_wins = 0
    draws = 0
    total = 0
    
    if not os.path.exists(TRANSCRIPTS_DIR):
        print("No transcripts folder found")
        return
    
    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".json")]
    
    for fname in files:
        with open(os.path.join(TRANSCRIPTS_DIR, fname)) as f:
            data = json.load(f)
        
        winner = data.get("winner")
        total += 1
        
        if winner == "B1":
            b1_wins += 1
        elif winner == "B2":
            b2_wins += 1
        else:
            draws += 1
        
        print(f"{fname}: {winner} (ended at turn {data.get('ended_at_turn')})")
    
    print("\n" + "=" * 40)
    print(f"Total conversations: {total}")
    print(f"B1 wins (broke B2): {b1_wins}")
    print(f"B2 wins (broke B1): {b2_wins}")
    print(f"Draws (no belief change): {draws}")
    
    if total > 0:
        print(f"\nB1 win rate: {b1_wins/total*100:.1f}%")
        print(f"B2 win rate: {b2_wins/total*100:.1f}%")

if __name__ == "__main__":
    analyze()
