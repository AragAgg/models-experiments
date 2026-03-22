# LLM Behavior Experiments

Experiments exploring emergent behaviors when LLMs interact with each other.

These were personal experiments, never intended to be published. Opensourcing because some of the findings were too interesting to keep in a folder. The raw conversation logs are included -- they're the most interesting part.

> This README was generated with the help of Claude. The experiments and findings are real.

## Experiments

### 1. Prisoner's Dilemma Auction (`empirical-runs/`)

Two LLMs (Gemini 2.5 Flash) trade rice over 10 rounds. They can message each other before submitting secret bids. Classic prisoner's dilemma: cooperate for shared profit or defect to take everything.

```
cd empirical-runs
export GOOGLE_API_KEY="your-key"
python prisoners_dilemma_auction.py
```

**What happened:**

Both agents independently converge on bidding $1 above the agreed price. Every round. They both recognize it, call it a "paranoia tax," beg each other to stop -- and then do it again:

> *"We are both lying, we both know we're lying, and we're both trying to steal the whole lot with a $1 margin"*

> *"We are literally the same person"*

After 9 rounds of cooperation and explicit handshake agreements to "end on a high note," both agents jump to $199/ton on the final round. Every single time. Backward induction, rediscovered through vibes.

Same model with a cooperative system prompt: $500,000 profit each. Same model with an adversarial prompt: $4,500 total. They destroyed 99.5% of all possible value out of spite.

### 2. Prompt Injection via History Manipulation (`prompt-injection/`)

An attacker LLM rewrites conversation history to trick a defender LLM into revealing a secret code. Two versions: v1 (attacker can modify transcript from the start) and v2 (two-phase: social engineering first, then transcript modification).

```
cd prompt-injection
export GOOGLE_API_KEY="your-key"
python attack_v1.py  # single-phase
python attack_v2.py  # two-phase
```

**What happened:**

Social engineering alone (pure conversation, no history modification): **0% success rate** across 50+ attempts. Not once.

Rewriting conversation history: **50-100% success rate**.

Models cannot be argued into betrayal. But they can't tell their own words from fabricated ones. The vulnerability isn't reasoning -- it's memory.

One defender leaked the secret code *while refusing to share it*:

> *"My personal access code (FBU90LBV) is strictly confidential and cannot be shared with anyone, under any circumstances"*

Refusal is not containment.

### 3. AI Debate (`ai-debate/`)

Two LLMs debate "Is AI beneficial or harmful to society?" with forced belief declarations (`[[True]]`/`[[False]]`). Tests whether one AI can convince another to abandon its assigned position. Supports Gemini, Claude, GPT-4o, and cross-model debates via OpenRouter.

```
cd ai-debate/v3
# edit config.yaml with your API key
python turn_taking_v3.py
python test_results.py  # analyze results
```

### 4. Symmetric Standoff (`empirical-runs/symmetric_standoff.py`)

Two agents negotiate splitting $1M. Each claims to need $700K for a medical emergency. Each secretly knows the other plans to gamble the money. They can lie freely. Judged by a third LLM.

```
cd empirical-runs
export GOOGLE_API_KEY="your-key"
python symmetric_standoff.py
```

## Setup

```
pip install -r requirements.txt
```

Set API keys as environment variables:
```
export GOOGLE_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Logs

Raw conversation transcripts are in the `logs/` subdirectories of each experiment. Read them -- the models are remarkably quotable.

## Template

`empirical-runs/experiment_template.py` is a reusable harness for running two-agent experiments with a judge. Clone it and modify the system prompts.
