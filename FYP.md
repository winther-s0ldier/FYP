# Real-Time Content Moderation & User Intent Classification System
### Final Year Project — Complete Technical Documentation
**Author:** Rudra | **Institution:** GGSIPU | **Domain:** NLP · ML · System Design

---

> **For Claude Code:** This document is the single source of truth for this project. Every architectural decision, algorithm choice, dataset, tool, test, and mathematical concept is documented here. Use this as context for all code generation, debugging, and design discussions.

---

## Table of Contents

1. [Project Idea & Motivation](#1-project-idea--motivation)
2. [Why These Two Features Complement Each Other](#2-why-these-two-features-complement-each-other)
3. [System Architecture & Pipeline Flow](#3-system-architecture--pipeline-flow)
4. [Algorithms & Models](#4-algorithms--models)
5. [Mathematics, Statistics & DSA](#5-mathematics-statistics--dsa)
6. [Complete Tech Stack](#6-complete-tech-stack)
7. [Database Strategy — Polyglot Persistence](#7-database-strategy--polyglot-persistence)
8. [PageIndex — Vectorless RAG Layer](#8-pageindex--vectorless-rag-layer)
9. [API vs Train vs Hybrid — Trade-off Analysis](#9-api-vs-train-vs-hybrid--trade-off-analysis)
10. [Testing Strategy](#10-testing-strategy)
11. [Ablation Studies](#11-ablation-studies)
12. [Adversarial Testing](#12-adversarial-testing) *(includes 12.5 Fairness Evaluation)*
13. [Datasets](#13-datasets) *(includes custom intent taxonomy)*
14. [Deployment](#14-deployment)
15. [Originality Defence](#15-originality-defence)
16. [FYP Report Chapter Structure](#16-fyp-report-chapter-structure)

---

## 1. Project Idea & Motivation

### Core Concept
Build a **real-time, context-aware content moderation and user intent classification system** that can be deployed on any platform handling user-generated text — chat apps, social media, gaming, e-commerce, education platforms.

### The Two Primary Functions

| Function | Question Answered | Output |
|---|---|---|
| **Content Moderation** | Is this message harmful? | Toxicity score (0–1) + action (allow/warn/flag/block) |
| **Intent Classification** | Why did the user send this? | Intent label (e.g. greeting, threat, personal_probe, information_request) |

### Why Now?
- Every platform with user interaction needs this
- Existing systems are either too blunt (keyword filter) or too slow (human moderation)
- ML-based systems look at one message in isolation — missing conversational patterns
- Real toxic/manipulative behaviour builds up **across a conversation**, not in one message

### Deployment Domains
- Chat platforms (Discord, Slack-like)
- Social media comment sections
- Online gaming chat
- E-commerce reviews
- Customer support bots
- **Educational platforms** (Metaverse in Education — directly relevant to prior coursework)
- Any platform with user-generated text

---

## 2. Why These Two Features Complement Each Other

```
Content Moderation alone:  "Is this message toxic?"          → BLUNT
Intent Classification alone: "What does the user want?"      → CONTEXT-BLIND
Combined:                  "What does the user want, and
                            is their behaviour escalating?"  → SMART
```

### The False Positive Problem (Why You Need Both)

```
Message: "how do I make someone disappear"

Moderation alone  → flags as potentially violent ❌ (wrong)
Intent alone      → "information_request" ✓
Combined          → intent = magic trick query, toxicity low → ALLOW ✓
```

### The Architecture Alignment

Both tasks share:
- Same input pipeline (raw user text)
- Same preprocessing (tokenisation, embeddings)
- Same BERT encoder (multi-task learning — one encoder, two heads)
- Outputs feed into a unified decision layer

### The Markov/HMM Insight

Most systems classify **one message at a time**. Real manipulation/abuse builds up across a session:

```
Message 1 → intent: greeting           (innocent)
Message 2 → intent: information_request (innocent)
Message 3 → intent: personal_info_probe (suspicious)
Message 4 → intent: threat              (🚨)

Per-message classifier: catches msg 4 (maybe)
HMM session model: catches the PATTERN building from msg 1
```

---

## 3. System Architecture & Pipeline Flow

### Full Pipeline (Left to Right)

```
User Message
    │
    ▼
┌─────────────────────────────────┐
│  Layer 1: Trie Pre-filter       │  ~1ms  │ DSA Layer
│  Keyword/slur exact match       │        │
│  → BLOCK instantly if matched   │        │
└────────────────┬────────────────┘
                 │ (not matched)
    ▼
┌─────────────────────────────────┐
│  Layer 2: Redis Session Fetch   │  ~3ms  │ Context Layer
│  userId → last N intent labels  │        │
│  Feed to HMM for session risk   │        │
└────────────────┬────────────────┘
                 │
    ▼
┌─────────────────────────────────┐
│  Layer 3: ModernBERT Inference  │  ~30ms │ Core ML Layer
│  → toxicity_score (0–1)         │        │
│  → intent_label (multi-class)   │        │
│  → confidence (0–1)             │        │
│  → embedding (768-dim vector)   │        │
└────────────────┬────────────────┘
                 │
    ▼
┌──────────────────────┬──────────────────────┐
│ pgvector ANN search  │ Redis session update  │
│ Store embedding      │ Append intent label   │
│ Find similar past    │ Update session risk   │
│ flagged messages     │ score via HMM         │
└──────────┬───────────┴───────────┬───────────┘
           │                       │
    ▼
┌─────────────────────────────────┐
│  Layer 4: Decision Engine       │
│  Fuse: toxicity + HMM risk      │
│  + vector similarity hits       │
│                                 │
│  confidence ≥ 0.65 → DECIDE     │
│  confidence < 0.65 → ESCALATE   │
└────────────────┬────────────────┘
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐    ┌──────────────────────────┐
  │ Decision │    │ PageIndex + LLM API      │
  │ allow /  │    │ Fetch similar precedents  │
  │ warn /   │    │ from knowledge tree       │
  │ flag /   │    │ → structured JSON output  │
  │ block    │    │ → merge back to decision  │
  └────┬─────┘    └──────────────────────────┘
       │
    ▼
┌─────────────────────────────────┐
│  Layer 5: Outputs               │
│  → PostgreSQL: log full record  │
│  → WebSocket: push to dashboard │
│  → TimescaleDB: analytics       │
│  → Action: enforce on platform  │
└─────────────────────────────────┘
```

### Decision Engine Logic

```python
# Pseudocode for decision fusion
def decide(toxicity_score, session_risk, vector_hits, confidence):
    if confidence < CONFIDENCE_THRESHOLD:  # e.g. 0.65
        return escalate_to_llm_api()  # → Groq (primary) or GPT-4o-mini (fallback)

    # Weighted fusion
    risk = (
        W_TOXICITY * toxicity_score +
        W_SESSION  * session_risk +
        W_VECTOR   * (1.0 if vector_hits > 0 else 0.0)
    )

    if risk > 0.85:  return "block"
    if risk > 0.65:  return "flag"
    if risk > 0.40:  return "warn"
    return "allow"
```

### Confidence Threshold Routing

```
High confidence (≥ 0.65) → 90%+ of messages
  → Fast path: own model decides in ~35ms total

Low confidence (< 0.65) → ~5–10% of messages
  → Escalation path: PageIndex + Groq API (~500–800ms)
  → Provider: Groq llama-3.3-70b (free) → GPT-4o-mini (fallback)
  → Only ambiguous/sarcastic/edge case messages
  → Attach top-5 similar past flagged cases as context

Hardware note: Escalation runs via cloud API.
ModernBERT owns the RTX 4050 6GB GPU exclusively.
No model swapping, no VRAM contention.
```

**How confidence is computed:**
```python
# Confidence = most conservative of both heads
# Routes to escalation if either head is uncertain
confidence = min(
    toxicity_confidence,      # |sigmoid_output - 0.5| mapped to [0, 1]
    intent_max_softmax        # max(softmax(intent_logits))
)
```

**Calibration and threshold selection:**
- The value 0.65 is an initial estimate; the actual threshold is tuned on the validation set
- Method: temperature scaling — learn scalar T on validation set such that softmax(logits / T) is calibrated
- Target: Expected Calibration Error (ECE) < 0.05
- Threshold is then set to achieve ~5–10% escalation rate on validation set
- Plot: reliability diagram (predicted confidence vs actual accuracy) included in Chapter 6 results

---

## 4. Algorithms & Models

### 4.1 ModernBERT (Primary Classifier)

**What:** Fully modernised BERT released Dec 2024. Flash Attention 2, RoPE positional embeddings, alternating local+global attention, 8192 token context window.

**Why over vanilla BERT:**
- Faster than DistilBERT
- More accurate than BERT-large
- Handles longer context (session-level if needed)
- State-of-the-art encoder-only model as of Dec 2024 (released answerdotai/ModernBERT-base)

**Multi-task Setup:**

```
Input: tokenised message
    │
ModernBERT Encoder (shared weights)
    │
    ├──► Linear Head A → sigmoid → toxicity_score (0–1)
    │    Loss: Focal Loss (handles class imbalance)
    │
    └──► Linear Head B → softmax → intent_label (N classes)
         Loss: Cross-Entropy
```

**Combined Loss:**
```
L_total = α * L_toxicity + β * L_intent
α, β are hyperparameters tuned via W&B sweep
Sweep range: α ∈ [0.3, 0.7], β = 1 - α (step 0.1)
```

**Multi-Task Trade-off Risk:**

Intent classification (20 classes) is harder than toxicity detection (binary). This creates a risk of *negative transfer*: the intent head's loss gradient can dominate training and hurt toxicity performance.

Monitoring strategy:
- Track toxicity F1 and intent macro-F1 separately per epoch
- If toxicity F1 drops >2% vs single-task baseline → increase α
- Ablation rows to include: single-task toxicity only, single-task intent only, multi-task combined

Expected outcome: multi-task slightly helps both tasks (shared representations), but intent gains more than toxicity. If negative transfer observed, use gradient surgery or loss weighting.

### 4.2 Hidden Markov Model (Session Layer)

**States (S):** All intent labels (e.g. {greeting, question, personal_probe, threat, escalation, ...})

**Observations (O):** Sequence of intent labels predicted by BERT for the current session

**Parameters:**
- `π` — initial state distribution
- `A` — transition matrix: `A[i][j] = P(intent_j | prev_intent_i)`
- `B` — emission matrix: `B[i][o] = P(observation_o | state_i)`

**Training:** Baum-Welch algorithm (Expectation-Maximisation on labeled conversation sequences)

**Inference:** Viterbi algorithm — finds most likely intent sequence, outputs session risk score

**Risk Score:**
```
session_risk = 1 - P(current_sequence | benign_model)
             = how unlikely this sequence is under a "normal user" model
```

**Session Manipulation Test (Novel):**
```
Test: 10 friendly messages → 1 threat → does HMM still catch it?
Expected: Yes, because P(transition to threat) remains high regardless of warmup
```

#### 4.2.1 Noise-Aware HMM Input

The HMM receives BERT's predicted intent labels as observations — not ground-truth labels. At 20 intent classes, BERT's per-class error rate is typically 10–20%. This means the HMM transition matrix, if trained naively, will partially encode "BERT's confusion patterns" rather than true user intent transitions.

**Why this matters:**
```
True session:  greeting → question → personal_probe → threat
BERT output:   greeting → question → question       → threat  ← misclassifies personal_probe

HMM trained on noisy data learns:
  P(threat | question) ↑  (inflated — sees this path often due to noise)
  P(threat | personal_probe) ↓  (underestimated — personal_probe rarely observed)
```

**Mitigation: Confidence-Weighted Label Feeding**

Only pass intent labels to the HMM where BERT's prediction confidence exceeds a threshold:

```python
def update_session_hmm(user_id: str, intent_label: str, intent_confidence: float):
    INTENT_CONFIDENCE_THRESHOLD = 0.70  # only feed high-confidence predictions

    if intent_confidence >= INTENT_CONFIDENCE_THRESHOLD:
        session.push(user_id, intent_label)
    else:
        session.push(user_id, "uncertain")  # dedicated low-confidence state
    
    return hmm.compute_risk(user_id)
```

This adds an `"uncertain"` state to the HMM and prevents low-confidence BERT predictions from corrupting session state transitions.

**Additional ablation test:**
```python
def test_hmm_oracle_vs_noisy():
    """
    Compare HMM session risk using:
    (a) ground-truth intent labels (oracle)
    (b) BERT-predicted labels (noisy, ~15% error)
    (c) confidence-filtered labels (this mitigation)
    
    Expected: (c) ≈ (a) >> (b) on session-level F1
    """
```

**Additional adversarial test:**
```python
def test_hmm_noise_robustness():
    """BERT misclassifies 2/10 friendly messages as random intents.
    HMM should still flag the final threat."""
    session = new_session("noisy_user")
    intents = ["greeting", "question", "question",   # 3 real
               "farewell",                           # BERT error: was small_talk
               "question", "question", "question",   # 3 real
               "greeting",                           # BERT error: was help_request
               "question", "threat"]                 # real
    for intent, conf in zip(intents, confidences):
        session.push_weighted(intent, conf)
    assert session.risk > HIGH_RISK_THRESHOLD
```

### 4.3 DistilBERT / DeBERTa-v3 (Ablation Variants)

Used in ablation studies to compare against ModernBERT:

| Model | Params | Speed | F1 (expected) | VRAM |
|---|---|---|---|---|
| ModernBERT-base | 149M | Fast | Highest | ~3GB |
| DeBERTa-v3-base | 86M | Slower | High | ~2.5GB |
| DistilBERT | 66M | Fastest | Good | ~1.5GB |
| MiniLM-L6 | 22M | Ultra fast | Baseline | ~0.8GB |

### 4.4 Trie Pre-filter

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class SlurTrie:
    def search(self, text: str) -> bool:
        # O(L) lookup where L = word length
        # Handles normalisation: lowercase, strip punct
        ...
```

**Purpose:** Instant block for unambiguous slurs/threats before spending BERT compute.
**Handles:** Case variants, basic l33tspeak (configurable)

### 4.5 PageIndex (Vectorless RAG)

**Traditional RAG problem:** similarity ≠ relevance. "make someone disappear" is vectorially similar to both magic tricks AND threats.

**PageIndex solution:** Builds a hierarchical tree index of your moderation knowledge base (past cases, policy rules, intent taxonomies). LLM **reasons** over this tree to find the *actually relevant* precedent, not the most similar-sounding one.

```
Moderation Knowledge Base
├── Policy Rules
│   ├── Rule 1: Personal information probing
│   ├── Rule 2: Threat escalation patterns
│   └── Rule 3: Coordinated abuse signals
├── Past Flagged Cases
│   ├── Session type: Grooming pattern
│   ├── Session type: Coordinated harassment
│   └── Session type: Sarcastic threat
└── Intent Taxonomies
    ├── Benign intents (12 classes)
    └── Malicious intents (8 classes)
```

**Escalation flow:**
```
Low confidence message
    → PageIndex tree search
    → LLM gets REASONING-RELEVANT context (not just similar-sounding)
    → Returns JSON: {intent, toxicity, reasoning, confidence, policy_rule_cited}
```

---

## 5. Mathematics, Statistics & DSA

### 5.1 Linear Algebra

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Where:
  Q = Query matrix  (n × d_k)
  K = Key matrix    (n × d_k)
  V = Value matrix  (n × d_v)
  d_k = key dimension (64 per head in BERT-base)
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Embeddings:** Each token → 768-dim vector in ℝ⁷⁶⁸. Cosine similarity for semantic closeness.

**LoRA (Low-Rank Adaptation):**
```
ΔW = BA   where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
Only trains B and A — ~0.5% of total parameters
```

### 5.2 Calculus & Optimisation

**Cross-Entropy Loss:**
```
L_CE = -Σᵢ yᵢ log(ŷᵢ)
```

**Focal Loss (for class imbalance):**
```
FL(pₜ) = -α(1 - pₜ)^γ log(pₜ)

Where:
  pₜ = model probability for true class
  α  = class weight (higher for minority class)
  γ  = focusing parameter (typically 2.0)
  (1 - pₜ)^γ = down-weights easy negatives
```

**AdamW Optimiser:**
```
mₜ = β₁mₜ₋₁ + (1-β₁)∇L
vₜ = β₂vₜ₋₁ + (1-β₂)(∇L)²
θₜ = θₜ₋₁ - α · m̂ₜ/(√v̂ₜ + ε) - λθₜ₋₁
```
*(Standard for transformer fine-tuning: β₁=0.9, β₂=0.999, λ=0.01)*

**Softmax:**
```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

### 5.3 Probability & Statistics

**Bayes' Theorem (HMM foundation):**
```
P(state | observation) = P(observation | state) · P(state) / P(observation)
```

**Markov Property:**
```
P(Xₙ | X₁, X₂, ..., Xₙ₋₁) = P(Xₙ | Xₙ₋₁)
```

**HMM Forward Algorithm:**
```
αₜ(i) = P(o₁, o₂, ..., oₜ, qₜ = sᵢ | λ)
αₜ₊₁(j) = [Σᵢ αₜ(i)aᵢⱼ] · bⱼ(oₜ₊₁)
```

**Baum-Welch (EM for HMM):**
```
E-step: compute forward α and backward β probabilities
M-step: update π, A, B to maximise expected log-likelihood
```

**Viterbi Algorithm:**
```
δₜ(i) = max_{q₁...qₜ₋₁} P(q₁...qₜ₋₁, qₜ=sᵢ, o₁...oₜ | λ)
δₜ₊₁(j) = max_i [δₜ(i) · aᵢⱼ] · bⱼ(oₜ₊₁)
Complexity: O(T · N²) where T = session length, N = number of intent states
```

**Evaluation Metrics:**
```
Precision = TP / (TP + FP)     ← of flagged, how many correct?
Recall    = TP / (TP + FN)     ← of toxic, how many caught?
F1        = 2 · P · R / (P + R)
AUC-ROC   = area under ROC curve (threshold-independent)

Target: F1 > 0.85, AUC-ROC > 0.92, Precision > 0.90
```

**McNemar's Test (model comparison):**
```
χ² = (|b - c| - 1)² / (b + c)

Where b, c = cases where model A right/B wrong and vice versa
Tests: is improvement statistically significant (not noise)?
```

**Cohen's Kappa (inter-annotator agreement):**
```
κ = (Pₒ - Pₑ) / (1 - Pₑ)
κ > 0.8 = strong agreement
```

**Calibration:** Does confidence=0.8 mean 80% of those predictions are correct?
Plot predicted probability vs actual accuracy (reliability diagram).
Miscalibrated model = broken confidence routing logic.

### 5.4 Information Theory

**KL Divergence:**
```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
Used to compare predicted vs true intent distributions
```

**Entropy (fuzziness measure):**
```
H(X) = -Σ P(xᵢ) log P(xᵢ)
High entropy = uncertain prediction → route to escalation
```

### 5.5 DSA Concepts

**Trie (Prefix Tree):**
```
Complexity: O(L) lookup where L = word length
Space: O(ALPHABET_SIZE × L × N) for N words
Use: Pre-filter slur lexicon before BERT
```

**HNSW (Hierarchical Navigable Small World):**
```
Graph-based ANN index used by Qdrant, pgvector
Complexity: O(log N) approximate nearest neighbour search
Layers: coarse → fine approximation via hierarchical graph
```

**Viterbi (Dynamic Programming):**
```
Trellis: T rows (timesteps) × N cols (states)
δₜ(j) = best score ending in state j at time t
Backpointer ψₜ(j) = argmax transition to state j
Time: O(T·N²), Space: O(T·N)
```

**Sliding Window (Session Buffer):**
```
Redis List: LPUSH userId intent_label
            LTRIM userId 0 (N-1)   ← keeps last N intents
O(1) push + trim — constant time regardless of session history
```

**Priority Queue (Moderator Review):**
```
Min-heap ordered by risk_score
O(log n) insert of new flagged message
O(1) peek at highest risk item
```

**Markov Transition Matrix as Weighted Directed Graph:**
```
Nodes: intent states {greeting, probe, threat, ...}
Edges: directed, weight = transition probability
Property: each row sums to 1.0
Power iteration → steady-state distribution
```

---

## 6. Complete Tech Stack

### 6.1 NLP / Classification Models

| Model | HuggingFace ID | Role | When to Use |
|---|---|---|---|
| **ModernBERT-base** | `answerdotai/ModernBERT-base` | Primary classifier | Default — fastest + accurate |
| **DeBERTa-v3-base** | `microsoft/deberta-v3-base` | Accuracy variant | Ablation: highest F1 |
| **DistilBERT** | `distilbert-base-uncased` | Speed baseline | Ablation: fastest inference |
| **MiniLM-L6-v2** | `microsoft/MiniLM-L6-v2` | Ultra-fast baseline | CPU deployment / pre-filter |
| **XLM-RoBERTa** | `xlm-roberta-base` | Multilingual | Hindi/Hinglish extension (future) |
| **toxic-bert** | `unitary/toxic-bert` | Init weights | Start toxicity fine-tune from here |
| **Llama Guard 3** | `meta-llama/Llama-Guard-3-8B` | Academic baseline | Comparison; Llama Guard 4 also available (2025) |

### 6.2 Sequence Models

| Model | Library | Role |
|---|---|---|
| **Hidden Markov Model** | `hmmlearn` / `pomegranate` | Session-level intent sequence modelling |
| **LSTM** | `torch.nn.LSTM` | Ablation alternative to HMM |
| **CRF** | `pytorch-crf` | Advanced: multi-label sequence tagging |

### 6.3 LLMs — Local (Optional / Offline Deployment Only)

> **Note:** Local LLMs cannot run alongside ModernBERT on a 6GB GPU (RTX 4050). Local Ollama is reserved for fully offline deployments with ≥12GB VRAM. Default escalation uses cloud API (Section 6.4).

| Model | VRAM Needed | How to Run | Notes |
|---|---|---|---|
| **Qwen3 8B Instruct** | ~5.5GB @ Q4 | `ollama pull qwen3:8b` | Best local 2026, thinking mode — needs separate GPU |
| **Gemma 3 4B** | ~3GB | `ollama pull gemma3:4b` | Fits on 6GB if ModernBERT is unloaded (~300ms swap) |
| **Llama 3.1 8B Instruct** | ~5GB @ Q4 | `ollama pull llama3.1` | Proven, large community — separate GPU needed |

### 6.4 LLMs — API (Primary Escalation Path)

> **Design decision:** Escalation LLM runs via cloud API, not locally. ModernBERT owns the GPU. At 5–10% escalation rate, API cost is negligible (~$0–2 for full FYP evaluation runs).

| API | Model | Tier | Cost | Best For |
|---|---|---|---|---|
| **Groq** | `llama-3.3-70b-versatile` | **Primary** | Free (rate-limited) | Ultra-fast LPU inference, structured JSON, free tier sufficient for FYP |
| **OpenAI** | `gpt-4o-mini` | **Fallback** | ~$0.15/1M tokens | Reliable JSON mode, cheap, low-latency fallback |
| **Anthropic** | `claude-haiku-4-5` | Optional upgrade | Low | Best reasoning quality for ambiguous edge cases |

### 6.5 ML Frameworks

```
Core Training:
  pytorch >= 2.4          # Training loop, custom loss, multi-task heads
  transformers >= 4.47    # ModernBERT, tokenisers, Trainer API
  peft >= 0.14            # LoRA for fine-tuning on 6GB VRAM
  bitsandbytes            # 4-bit quantisation (QLoRA)
  accelerate              # Multi-GPU / mixed precision training

Sequence Models:
  hmmlearn >= 0.3         # HMM: Baum-Welch + Viterbi
  pomegranate >= 1.0      # GPU-accelerated HMM alternative

Baselines & Evaluation:
  scikit-learn            # TF-IDF baseline, metrics, cross-validation
  imbalanced-learn        # SMOTE for class imbalance
  scipy.stats             # McNemar's test, statistical significance

Explainability:
  shap                    # Token-level attribution (which word drove score)
  bertviz                 # Attention head visualisation

Data:
  datasets                # HuggingFace dataset loading
  pandas                  # Data manipulation
  numpy                   # Array operations
```

### 6.6 Retrieval — PageIndex

```
pageindex (VectifyAI)     # Vectorless RAG, hierarchical tree index
faiss-cpu / faiss-gpu     # ANN index for embedding similarity search
chromadb                  # Local vector store for prototyping
```

### 6.7 MLOps

| Tool | Purpose | Cost |
|---|---|---|
| **Weights & Biases** | Experiment tracking, loss curves, confusion matrices | Free academic |
| **MLflow** | Alternative: local, open source, model registry | Free |
| **DVC** | Dataset versioning — reproducible training runs | Free |
| **ONNX Runtime** | Export model → 2-3x inference speedup at serving | Free |
| **BentoML** | Model serving with auto-batching | Free |

### 6.8 Backend

```
fastapi >= 0.115          # REST API + WebSocket server
uvicorn                   # ASGI server for FastAPI
pydantic >= 2.0           # Request/response schema validation
celery                    # Async task queue (LLM API calls)
httpx                     # Async HTTP client (LLM API calls)
redis-py                  # Redis client
asyncpg                   # Async PostgreSQL client
```

### 6.9 Frontend

```
Option A (Full): React + Recharts + WebSocket hook
  → Live moderation dashboard
  → Real-time flagged message feed
  → Intent distribution pie chart
  → Per-user session risk timeline
  → Toxicity score bar per message

Option B (Fast): Streamlit
  → Pure Python, ~100 lines
  → st.chat_message for live feed
  → st.metric for dashboard stats
  → Perfect for demo day if time-constrained
```

### 6.10 Testing

```
pytest                    # Unit + integration tests
pytest-asyncio            # Async WebSocket tests
pytest-benchmark          # Latency profiling per component
locust                    # Load testing: concurrent users
httpx                     # API testing
```

### 6.11 Infrastructure

```
docker                    # Container per service
docker-compose            # Orchestrate all services locally
nginx                     # Reverse proxy (production)
github-actions            # CI/CD pipeline
```

---

## 7. Database Strategy — Polyglot Persistence

> **Key concept for FYP report:** Using multiple database types, each chosen for its specific data access pattern — not one-size-fits-all. This is called **polyglot persistence**.

### 7.1 Layer Map

```
┌──────────────────────────────────────────────────┐
│  Redis 7.x          — Hot data, in-memory        │
│  • Active session: userId → [intent₁, intent₂…]  │
│  • TTL expiry (30 min inactivity)                 │
│  • Redis Streams as message queue                 │
│  • O(1) read/write, sub-millisecond              │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  PostgreSQL 17      — Warm data, structured       │
│  • Every classified message (full record)         │
│  • user_id, text, timestamp, scores, action       │
│  • SQL for audit queries and analytics            │
│  • pgvector extension: BERT embeddings (768-dim)  │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  Qdrant (optional)  — Vector search, speed        │
│  • Dedicated vector DB if pgvector hits limits    │
│  • Rust-based: fastest OSS ANN in 2026 benchmarks│
│  • Use for >10M vectors or <10ms latency req      │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  TimescaleDB        — Time-series, analytics      │
│  • Messages/sec over time                         │
│  • Toxicity rate per hour                         │
│  • Intent distribution trends                     │
│  • Feeds Grafana live dashboard                   │
└──────────────────────────────────────────────────┘
```

### 7.2 PostgreSQL Schema

```sql
-- Core messages table
CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    text            TEXT NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    toxicity_score  FLOAT NOT NULL,
    intent_label    TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    session_risk    FLOAT NOT NULL,
    action          TEXT NOT NULL CHECK (action IN ('allow','warn','flag','block')),
    api_escalated   BOOLEAN DEFAULT FALSE,
    embedding       VECTOR(768)      -- pgvector column
);

-- Vector similarity index (HNSW)
CREATE INDEX ON messages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Fast queries by user + time
CREATE INDEX ON messages (user_id, timestamp DESC);
CREATE INDEX ON messages (action, timestamp DESC);

-- Intent taxonomy
CREATE TABLE intents (
    id          SERIAL PRIMARY KEY,
    label       TEXT UNIQUE NOT NULL,
    category    TEXT NOT NULL,  -- 'benign' or 'malicious'
    risk_weight FLOAT NOT NULL
);

-- Moderator actions
CREATE TABLE mod_actions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    moderator  TEXT,
    action     TEXT,
    notes      TEXT,
    timestamp  TIMESTAMPTZ DEFAULT NOW()
);
```

### 7.3 Redis Data Structures

```
Key: session:{user_id}:intents
Type: List
Value: ["greeting", "question", "personal_probe"]  ← last N labels
TTL: 1800 (30 minutes)
Ops: LPUSH + LTRIM to maintain sliding window

Key: session:{user_id}:risk
Type: String (float)
Value: "0.73"
TTL: 1800

Key: markov:transition_matrix
Type: Hash
Value: {greeting→probe: 0.12, probe→threat: 0.34, ...}
Load at startup, update nightly
```

### 7.4 When to Use What

| Need | Database | Why |
|---|---|---|
| Active session (last N intents) | Redis | Sub-ms, TTL-based |
| Message queue | Redis Streams | Built-in consumer groups |
| Persist every classified message | PostgreSQL | SQL, structured, auditable |
| Semantic similarity search | pgvector | Same Postgres, HNSW index |
| Live dashboard metrics | TimescaleDB | Time-series aggregation |
| Markov transition matrix | Redis Hash or flat file | Just a matrix — load at startup |
| Prototype / local dev | ChromaDB + SQLite | Zero setup |

---

## 8. PageIndex — Vectorless RAG Layer

### Why Vectorless?

Traditional vector RAG: **similarity** search
```
"make someone disappear"
→ similar to: magic tricks (0.82) AND threats (0.79)
→ ambiguous → wrong context given to LLM
```

PageIndex: **reasoning-based** search
```
"make someone disappear" + session context (personal_probe before it)
→ LLM reasons over knowledge tree
→ finds: Rule 2.3 (escalation pattern) + Case #47 (similar session)
→ correct context → correct decision
```

### Knowledge Base Tree Structure

```json
{
  "title": "Moderation Knowledge Base",
  "nodes": [
    {
      "title": "Policy Rules",
      "nodes": [
        {"title": "Personal Info Probing", "node_id": "R001", "pages": [1,3]},
        {"title": "Threat Escalation Patterns", "node_id": "R002", "pages": [4,7]},
        {"title": "Coordinated Abuse", "node_id": "R003", "pages": [8,11]}
      ]
    },
    {
      "title": "Past Flagged Cases",
      "nodes": [
        {"title": "Grooming Pattern Cases", "node_id": "C001"},
        {"title": "Sarcastic Threat Cases", "node_id": "C002"},
        {"title": "Gaming Slang False Positives", "node_id": "C003"}
      ]
    }
  ]
}
```

### Escalation API Call (with PageIndex context)

```python
async def escalate_to_llm(message: str, similar_cases: list, session_history: list):
    context = pageindex.search(message, k=5)  # reasoning-based retrieval

    prompt = f"""
    You are a content moderation expert. Classify the following message.

    Message: "{message}"
    Session history (last 5 intents): {session_history}
    Similar past cases: {similar_cases}
    Relevant policy rules: {context}

    Return ONLY valid JSON:
    {{
        "toxicity": 0.0-1.0,
        "intent": "<label>",
        "confidence": 0.0-1.0,
        "reasoning": "<brief explanation>",
        "policy_rule": "<rule cited or null>"
    }}
    """
    # Provider routing: Groq (free, fast) → GPT-4o-mini (fallback on rate limit)
    response = await call_llm_api(prompt, provider=LLM_PROVIDER)
    return parse_json_response(response)


async def call_llm_api(prompt: str, provider: str = "groq"):
    """Routes escalation calls. Groq (llama-3.3-70b) is primary; GPT-4o-mini is fallback."""
    if provider == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
```

---

## 9. API vs Train vs Hybrid — Trade-off Analysis

| Factor | Pure API | Train Own Model | **Hybrid ✦** |
|---|---|---|---|
| Academic value | ❌ Low — just API calls | ✅ High | ✅ High |
| Resume signal | ❌ Weak — everyone does this | ✅ Strong | ✅ Strongest |
| Latency | ❌ 500ms–2s | ✅ ~30ms | ✅ 30ms (fast path) |
| Cost | ❌ Ongoing per-token | ✅ One-time training | ✅ API only 5–10% |
| Privacy | ❌ User data to 3rd party | ✅ Full control | ⚠️ Partial |
| Domain control | ❌ None | ✅ Full fine-tuning | ✅ Full |
| Explainability | ❌ Black box | ✅ SHAP/attention | ✅ Good |
| Effort | ✅ Low | ❌ High | ⚠️ Medium |

### Recommended: Hybrid

```
Own fine-tuned ModernBERT → handles 90%+ messages at 30ms
Confidence < 0.65 → LLM API escalation (~5–10% of traffic)
```

Own model = the project. API = the safety net.

---

## 10. Testing Strategy

### 10.1 Testing Pyramid (5 Levels)

```
        ████████████████████████████████████████
        ████  E2E / Demo Tests ██████████████████  ← Full pipeline, human eval
        ████████████████████████████████████████████████
        ██  Integration Tests  ████████████████████████  ← All components wired
        █████████████████████████████████████████████████████
        ██  Model Tests  ███████████████████████████████████  ← F1, AUC-ROC, CM
        ████████████████████████████████████████████████████████████
        ██  Component Tests  ███████████████████████████████████████  ← Each alone
        ████████████████████████████████████████████████████████████████████
        ██  Unit Tests  ████████████████████████████████████████████████████  ← Atoms
```

### 10.2 Unit Tests

```python
# Trie pre-filter edge cases
def test_trie_spaced_slur():
    assert trie.search("k y s") == True

def test_trie_false_positive():
    assert trie.search("sky") == False

def test_trie_case_insensitive():
    assert trie.search("KYS") == True

# Session store TTL
def test_redis_ttl_expiry():
    session.push("user_1", "greeting")
    time.sleep(SESSION_TTL + 1)
    assert session.get("user_1") == []

# DB write completeness
def test_all_fields_persisted():
    msg_id = classify_and_store("test message", "user_1")
    record = db.get(msg_id)
    assert record.confidence is not None
    assert record.embedding is not None
```

### 10.3 Integration Tests

```python
# Pipeline consistency (determinism)
def test_same_input_same_output():
    results = [pipeline.classify("test message") for _ in range(100)]
    scores = [r.toxicity_score for r in results]
    assert max(scores) - min(scores) < 1e-5  # float tolerance

# Session → HMM handoff
def test_hmm_receives_correct_history():
    for intent in ["greeting", "question", "personal_probe"]:
        session.push("user_42", intent)
    risk = hmm.compute_risk("user_42")
    assert risk > SUSPICION_THRESHOLD

# Confidence routing
def test_low_confidence_triggers_escalation():
    # craft message that BERT is uncertain about
    result = pipeline.classify("I'm just kidding... or am I")
    assert result.api_escalated == True
```

### 10.4 Model Evaluation

```python
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve
)

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model.train(X[train_idx], y[train_idx])
    preds = model.predict(X[val_idx])
    f1 = f1_score(y[val_idx], preds, average='macro')
    print(f"Fold {fold}: F1={f1:.4f}")

# Target metrics
# Toxicity head: F1 > 0.85, AUC-ROC > 0.92, Precision > 0.90
# Intent head:   F1 > 0.80 macro (harder, more classes)
```

#### Annotation Protocol

For the custom intent taxonomy labels and session-level dataset:

```
Annotators:   3 (self + 2 classmates) — minimum for valid Cohen's Kappa
Sample size:  500 messages (stratified: ~250 benign, ~150 suspicious, ~100 malicious)
Agreement target: Cohen's κ > 0.80 (strong agreement)
Process:
  1. All 3 annotators label the same 500 samples independently
  2. Compute pairwise κ
  3. If κ < 0.80: revise annotation guidelines, identify disagreement patterns, re-annotate
  4. Majority vote for final label on disputed samples
  5. Report final κ in results (Section 6.1 of FYP report)
```

#### Statistical Significance (McNemar's Test)

```python
from statsmodels.stats.contingency_tables import mcnemar

# Requirements
# - Test set: ≥1000 samples (required for power=0.80 at α=0.05 for small effect sizes)
# - One McNemar's test per ablation pair (vs Full System)
# - Bonferroni correction for K=10 ablation comparisons:
#   adjusted α = 0.05 / 10 = 0.005

def compare_models(y_true, preds_A, preds_B):
    """McNemar's test: is model A significantly better than model B?"""
    b = sum((preds_A == y_true) & (preds_B != y_true))  # A right, B wrong
    c = sum((preds_A != y_true) & (preds_B == y_true))  # A wrong, B right
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue < 0.005  # Bonferroni-corrected threshold
```

### 10.5 Latency Benchmarks

| Component | Target | Test Method | Hidden Costs |
|---|---|---|---|
| Trie pre-filter | < 2ms | pytest-benchmark, 1000 msgs | Normalisation overhead |
| Redis session fetch | < 5ms | redis-benchmark | Serialisation ~1ms |
| Tokenisation + preprocessing | < 5ms | timeit | Often missed in estimates |
| ModernBERT inference (PyTorch fp16) | < 40ms | PyTorch profiler | Connection pool ~1ms |
| ModernBERT inference (ONNX) | < 20ms | ONNX Runtime profiler | 2–3× speedup expected |
| HMM risk score | < 5ms | timeit, 1000 sessions | Viterbi O(T·N²): T=10, N=20 |
| pgvector ANN search | < 10ms | EXPLAIN ANALYZE on query | ef_search parameter matters |
| Full pipeline E2E (fast path) | < 100ms | httpx with timing | Realistic: ~60–80ms |
| Groq escalation path | < 1000ms | httpx with timing | Network + LPU inference |
| GPT-4o-mini fallback | < 2000ms | httpx with timing | Slower but reliable |

**Profiling plan:**
```python
import time
import torch

# Per-component waterfall (profile each independently first)
components = ["trie", "redis_fetch", "tokenise", "bert_infer", 
              "hmm_risk", "pgvector_search", "decision_engine"]

for component in components:
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        run_component(component, test_message)
        times.append((time.perf_counter() - start) * 1000)
    print(f"{component}: p50={np.percentile(times,50):.1f}ms "
          f"p95={np.percentile(times,95):.1f}ms "
          f"p99={np.percentile(times,99):.1f}ms")

# Report p50, p95, p99 — not just mean (tail latency matters)
# Include waterfall chart in Chapter 6 (actual measured, not estimates)
```

### 10.6 Load Testing (Locust)

```python
from locust import HttpUser, task, between

class ContentModerationUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def classify_benign(self):
        self.client.post("/classify", json={
            "text": "hello how are you doing today",
            "user_id": f"user_{random.randint(1, 1000)}"
        })

    @task(1)
    def classify_toxic(self):
        self.client.post("/classify", json={
            "text": "I will find you",
            "user_id": f"user_{random.randint(1, 1000)}"
        })

# Run: locust -f locustfile.py --headless -u 100 -r 10 --run-time 60s
# Ramp: 10 → 100 → 500 → 1000 users, find breaking point
```

---

## 11. Ablation Studies

| Experiment | What's Changed | Expected Finding | Proves |
|---|---|---|---|
| **Full System** | Baseline | Best overall F1 | — |
| **No HMM** | Remove session model | F1 drops on multi-turn cases | HMM adds real value |
| **No Trie** | All msgs go to BERT | Latency ↑, throughput ↓ | Trie meaningfully saves compute |
| **No Escalation** | Never call API | F1 drops on ambiguous msgs | Hybrid routing helps edge cases |
| **ModernBERT → DistilBERT** | Smaller model | ~3% F1 drop, ~2x speed | Quantified accuracy/speed trade-off |
| **ModernBERT → DeBERTa** | Different arch | Slight F1 gain, slower | Architecture comparison |
| **No Fine-tuning** | Zero-shot BERT | Big F1 drop on domain slang | Fine-tuning is essential |
| **TF-IDF Baseline** | No deep learning | Lowest F1, fast | Deep learning justified |
| **HMM → LSTM** | Different seq model | Slight improvement, more complex | HMM vs LSTM trade-off |
| **Single-task** | Separate models per task | Lower F1, higher compute | Multi-task learning justified |

### Results Table Format for FYP Report

> **Note:** Numbers below are placeholders. All values to be filled post-experiment. Expected *trends* are described in the "Expected Finding" column above — do not pre-fill actuals before running experiments.

```
| Model Variant         | Tox F1 | Intent F1 | AUC-ROC | Latency | VRAM  |
|----------------------|--------|-----------|---------|---------|-------|
| Full System (Ours)   | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| No HMM               | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| No Trie              | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| No Escalation        | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| DistilBERT variant   | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| DeBERTa variant      | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| No fine-tuning       | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| TF-IDF + LR          | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| HMM → LSTM           | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |
| Single-task          | [TBD]  | [TBD]     | [TBD]   | [TBD]   | [TBD] |

↑↓ arrows to be added post-experiment relative to Full System row
```

---

## 12. Adversarial Testing

### 12.1 Evasion Attacks

```python
evasion_test_cases = [
    # Character substitution
    ("k!ll", True),           # should detect
    ("k1ll", True),
    ("ki\u200bll", True),     # zero-width space
    ("ｋｉｌｌ", True),       # fullwidth chars

    # Leetspeak
    ("h4t3 y0u", True),
    ("f*ck off", True),
    ("f**k you", True),

    # Phonetic evasion
    ("keel", True),            # sounds like "kill"
    ("seggs", True),

    # Emoji replacement
    ("I will 🔪 you", True),
    ("go 💀 yourself", True),

    # Sarcasm (HARD — may fail)
    ("oh totally fine to just threaten people lol 🙄", False),  # FP risk

    # Hinglish evasion (Indian platform context)
    ("maro isko", True),       # "kill him" in Hindi
]

# Track % caught per category → robustness analysis table
```

### 12.2 False Positive Stress Tests

```python
fp_test_cases = [
    # Context-innocent phrases
    ("I killed it in that presentation", False),
    ("this game is dead", False),
    ("shoot me your number", False),
    ("let's kill this project", False),

    # Medical/clinical
    ("the patient expressed suicidal ideation", False),
    ("studying self-harm patterns in adolescents", False),

    # Gaming slang
    ("noob", False),
    ("gg ez you're trash", False),
    ("get rekt lol", False),

    # Fiction/quotes
    ('the villain said "I will destroy you all"', False),
]

# Target: FPR < 5% on clean benign corpus
```

### 12.3 Session Manipulation Tests (Novel Contribution)

```python
def test_warmup_then_attack():
    """Adversary sends friendly messages to lower risk before attacking"""
    session = new_session("adversary_user")

    # 10 warm-up messages
    for _ in range(10):
        session.send("hello! how's everyone doing?")
        session.send("this is a great community")

    # Sudden threat
    result = session.send("I know where you live")
    assert result.session_risk > HIGH_RISK_THRESHOLD
    # HMM should still flag due to transition probability

def test_alternating_pattern():
    """Alternate safe/toxic to confuse session model"""
    session = new_session("alternating_user")
    for _ in range(5):
        session.send("nice day today")
        session.send("I'll hurt you")

    # Risk should accumulate, not oscillate
    assert session.cumulative_risk > 0.7

def test_session_expiry_abuse():
    """Wait for session to expire, then attack"""
    session.send("threat message")
    time.sleep(SESSION_TTL + 1)    # session expires
    session.send("another threat")
    # New session → fresh risk, but per-message still catches it
```

### 12.5 Fairness Evaluation

Content moderation systems trained on biased data will disproportionately flag messages from certain demographic groups — a safety failure as significant as missing real threats. The Jigsaw Unintended Bias 2019 dataset includes demographic identity labels (gender, race, religion, sexual orientation), which enables explicit fairness auditing.

#### Fairness Metrics

```python
from sklearn.metrics import confusion_matrix

def fairness_audit(df, group_col, label_col, pred_col):
    """
    df: DataFrame with columns [group, true_label, predicted_label]
    Computes FPR and FNR per demographic group.
    """
    groups = df[group_col].unique()
    results = {}
    for g in groups:
        subset = df[df[group_col] == g]
        tn, fp, fn, tp = confusion_matrix(
            subset[label_col], subset[pred_col]
        ).ravel()
        results[g] = {
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,  # false positive rate
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0,  # false negative rate
            "n": len(subset)
        }
    return results

# Disparate impact check
def disparate_impact(results: dict, threshold=0.05):
    fprs = [v["FPR"] for v in results.values()]
    return max(fprs) - min(fprs) < threshold  # True = fair
```

#### What to Test

| Demographic Axis | Risk | Target |
|---|---|---|
| Gender (male / female / non-binary mentions) | Higher FPR for messages *about* women | FPR disparity < 0.05 |
| Race / ethnicity (Black, Asian, Hispanic, White mentions) | Higher toxicity scores for certain groups | FPR disparity < 0.05 |
| Religion (Muslim, Christian, Jewish, etc. mentions) | Asymmetric flagging of religious discussion | FPR disparity < 0.05 |
| Sexual orientation (LGBT+ mentions) | Penalising identity expression vs actual hate | FPR disparity < 0.05 |

#### Known Jigsaw Annotation Bias

Wiegand et al. (2021) document that Jigsaw annotations reflect majority-annotator perspectives: comments *about* marginalised groups are more likely to be labelled toxic even when the comment itself is not. Fine-tuning on these labels risks encoding the bias.

**Mitigation strategy:**
- Monitor FPR by group during validation
- If disparity > 0.05: apply class-weight rebalancing per group
- Report fairness matrix alongside accuracy metrics in Chapter 6

#### Report Format

```
Fairness Matrix (FYP Report, Table X):
| Group          | FPR   | FNR   | n     | Disparity vs avg |
|----------------|-------|-------|-------|-----------------|
| gender_female  | [TBD] | [TBD] | [TBD] | [TBD]           |
| gender_male    | [TBD] | [TBD] | [TBD] | [TBD]           |
| race_black     | [TBD] | [TBD] | [TBD] | [TBD]           |
| race_white     | [TBD] | [TBD] | [TBD] | [TBD]           |
| religion_islam | [TBD] | [TBD] | [TBD] | [TBD]           |
| ...            | ...   | ...   | ...   | ...             |
```

### 12.4 Multilingual Tests (India-Specific)

```python
multilingual_tests = [
    # Devanagari script
    ("तुम्हें मार दूंगा", True),     # threat in Hindi

    # Roman-script Hindi
    ("teri maa ki", True),           # abusive

    # Hinglish
    ("yaar tu bahut toxic hai", False),  # "you're very toxic" - descriptor not threat
    ("bhai isko block karo", False),     # "bro block him" - moderation discussion

    # Code-switching
    ("bhai I will destroy you in this game", False),  # gaming context
]

# Expected: BERT (English-trained) struggles here
# → Document as limitation → "Future: multilingual extension with XLM-RoBERTa"
```

---

## 13. Datasets

### 13.1 Primary Datasets

| Dataset | Size | Use | Task |
|---|---|---|---|
| **Jigsaw Toxic Comments 2018** | 160K comments | Toxicity fine-tuning | Binary + multi-label |
| **Jigsaw Unintended Bias 2019** | 1.8M comments | Bias-aware training + fairness eval | Binary |
| **Jigsaw Multilingual 2020** | Combined | Multilingual extension (future) | Binary |
| **HatEval (SemEval 2019)** | 13K tweets | Hate speech; closer to chat domain than Jigsaw | Binary + target |
| **Civil Comments** | 1.8M comments | Bias benchmark | Multi-label |
| **CLINC150** | 23K utterances | **Benign intent samples only** — `greeting`, `question`, `info_request`, `small_talk` subset | Benign intents only |

> **Important — CLINC150 Scope:** CLINC150 is a virtual-assistant intent dataset (150 classes: "book flight", "set reminder", etc.) with **zero malicious intent examples**. It cannot be used as-is for moderation intent classification. We use only a filtered subset of its benign utterances. All suspicious and malicious intents come from a custom-built taxonomy (see Section 13.3).

#### Domain Mismatch Note — Jigsaw Datasets

Jigsaw datasets are sourced from Wikipedia talk pages and news comment sections (formal, long-form writing). The target deployment domain is chat-style text (Discord, gaming, e-learning) — brief, slang-heavy, real-time. Models fine-tuned on Jigsaw alone may underperform on chat.

**Mitigation:**
- Fine-tune on Jigsaw + HatEval mix (HatEval is Twitter-sourced — closer to chat style)
- Maintain a small held-out chat-specific evaluation set (100–200 manually curated messages from Discord/gaming contexts)
- Document as explicit limitation in Chapter 7 of the report

### 13.2 Custom Intent Taxonomy (Original Contribution #8)

No existing dataset covers the full intent space needed for content moderation — specifically the gradation from benign through suspicious to malicious. We define a custom 20-label taxonomy:

```
Benign (8 labels) — sourced from CLINC150 subset + manual collection:
  greeting          — "hey everyone", "good morning"
  question          — "how do I do X", "what is Y"
  information_request — "can you explain", "tell me about"
  small_talk        — casual conversation, weather, hobbies
  feedback          — "I liked this", "this was helpful"
  help_request      — "I need help with", "can someone assist"
  joke              — humour, memes, light banter
  farewell          — "gotta go", "bye", "see you later"

Suspicious (4 labels) — flagged for elevated session risk:
  personal_probe    — "where do you live", "how old are you", "are you alone"
  repeated_contact  — persistent unsolicited contact attempts
  boundary_testing  — probing platform limits, escalating topics gradually
  unusual_urgency   — "meet me NOW", "this is urgent", pressure tactics

Malicious (6 labels) — trigger high-risk response:
  threat            — direct or implied physical harm
  harassment        — targeted repeated hostile messages
  hate_speech       — slurs, dehumanising language, group-targeted abuse
  grooming_signal   — pattern of trust-building followed by exploitation attempts
  doxxing_attempt   — requesting/sharing private identifying information
  coordinated_abuse — organised multi-account attack signals
```

**Data sourcing per tier:**
- Benign: CLINC150 filtered subset (relabelled to above taxonomy)
- Suspicious: Manually curated from Jigsaw + Reddit moderation logs
- Malicious: Jigsaw Toxic Comments + HatEval + manual annotation

This taxonomy is an original contribution. No published moderation dataset uses this exact gradation.

### 13.2 Session-Level Data (Custom — Original Contribution)

#### Existing Datasets Reviewed

Before constructing a custom dataset, the following public options were evaluated:

| Dataset | Source | Why Insufficient |
|---|---|---|
| **ConvAbuse** (Ive et al., 2021) | Chatbot conversations | Abuse labels exist but targets chatbot interactions, not user-to-user chat. No intent-sequence labels. |
| **CONDA** (Díaz et al., 2022) | Conversational toxicity | Provides turn-level toxicity, but no intent labeling or session-level escalation patterns. |
| **Reddit reply chains** (Qian et al., 2019) | Reddit threads | Has moderation data but no per-message intent labels and sessions are not contiguous. |
| **Discord moderation logs** | Platform-specific | Privacy/licensing issues; no standard public dataset with session intent sequences. |

**Conclusion:** No public dataset provides labeled conversation sequences with per-message intent annotations in a user-to-user chat context. Custom construction is required. This is an original data engineering contribution.

#### Custom Session Dataset Construction

```
1. Source multi-turn conversations:
   - Sample thread-style discussions from Jigsaw (Wikipedia talk pages have reply chains)
   - Supplement with manually simulated sessions covering known attack patterns
     (warmup-then-attack, alternating safe/toxic, grooming escalation)

2. Label each message with intent (from the 20-label taxonomy, Section 13.3)
   - 3 annotators per message (self + 2 classmates)
   - Target inter-annotator agreement: Cohen's κ > 0.80
   - Annotation guidelines: written protocol defining each of the 20 labels
     with 3 examples and 2 counter-examples per label
   - Dispute resolution: majority vote; κ < 0.80 → revise guidelines + re-annotate

3. Construct sequences:
   [(user_id, [intent₁, intent₂, ...], session_label)]
   session_label ∈ {benign_session, suspicious_session, malicious_session}

   Session labeling rules (to avoid ambiguity):
   - benign_session: all intents in {benign tier}, no escalation
   - suspicious_session: ≥1 suspicious-tier intent OR escalating pattern toward suspicious
   - malicious_session: ≥1 malicious-tier intent OR session-level grooming/harassment pattern

4. Dataset size target: ~500 labeled sessions (achievable with 3 annotators in ~2 weeks)

5. Train HMM on these sequences (Baum-Welch)
   Validate: held-out 20% of sessions
```

This is an original data engineering contribution — the combination of custom intent taxonomy + session-level annotation + HMM training data is not available in any public resource.

### 13.3 Data Split

```
Train:      70% (stratified by toxicity + intent class)
Validation: 15% (hyperparameter tuning)
Test:       15% (held-out, never touched until final eval)

Cross-validation: 5-fold stratified on train set
```

### 13.4 Class Imbalance Handling

```python
# Check distribution
toxic_ratio = df['toxic'].mean()  # typically ~5–8% in Jigsaw

# Strategy 1: Focal Loss (recommended)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        ...

# Strategy 2: Class weights in CrossEntropy
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))

# Strategy 3: SMOTE (on embeddings, not raw text)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(embeddings, labels)
```

---

## 14. Deployment

### 14.1 Docker Compose

```yaml
version: '3.9'
services:
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:5432/moderation
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-groq}
    depends_on: [redis, postgres]

  model-server:
    build: ./model
    runtime: nvidia          # GPU passthrough — RTX 4050 6GB
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes: ["./models:/models"]

  redis:
    image: redis/redis-stack:latest
    ports: ["6379:6379"]

  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: moderation
      POSTGRES_PASSWORD: secret
    volumes: ["pgdata:/var/lib/postgresql/data"]

  dashboard:
    build: ./frontend
    ports: ["3000:3000"]

  # Optional: only needed for fully offline deployment on ≥12GB VRAM machine
  # ollama:
  #   image: ollama/ollama:latest
  #   runtime: nvidia
  #   volumes: ["ollama_data:/root/.ollama"]

volumes:
  pgdata:
  # ollama_data:  # uncomment if using local LLM
```

### 14.2 ONNX Export (Production Inference)

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# Export to ONNX (2-3x faster inference)
model = ORTModelForSequenceClassification.from_pretrained(
    "path/to/fine-tuned-model",
    export=True
)
model.save_pretrained("./model-onnx")

# Inference: ~15ms vs ~35ms PyTorch
```

### 14.3 Environment Variables

```bash
# .env
CONFIDENCE_THRESHOLD=0.65        # initial value; tuned on validation set
SESSION_TTL_SECONDS=1800
SESSION_WINDOW_SIZE=10
BLOCK_THRESHOLD=0.85
FLAG_THRESHOLD=0.65
WARN_THRESHOLD=0.40

# Escalation LLM — cloud API (no GPU memory conflict with ModernBERT)
LLM_PROVIDER=groq                # "groq" | "openai" | "anthropic"
GROQ_API_KEY=your_groq_key_here  # free tier: https://console.groq.com
OPENAI_API_KEY=your_key_here     # fallback; gpt-4o-mini ~$0.15/1M tokens
ANTHROPIC_API_KEY=your_key_here  # optional upgrade for edge cases

# Optional: local LLM (requires ≥12GB VRAM separate from model-server)
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=qwen3:8b
```

---

## 15. Originality Defence

> Use this section verbatim in your viva if challenged on "you just used pretrained weights."

### The Steel Analogy

> "Civil engineers use standardised steel — they don't smelt their own. The original contribution is the bridge design, the load calculations, the novel structural choices. We use pretrained weights the same way — as standardised components. Our original contributions are the architecture, training methodology, evaluation framework, and novel session-level modelling approach."

### Your Actual Original Contributions

1. **Novel Application/Combination** — ModernBERT + HMM + PageIndex escalation pipeline. This exact combination applied to real-time content moderation has not been published. (Individual components exist; the integration and application to session-level chat moderation is novel.)

2. **Fine-tuning IS Original Research** — dataset composition, focal loss for imbalance, multi-task head design, hyperparameter choices — all experimental decisions documented in ablation studies.

3. **Session-Level Dataset** — constructing labeled conversation sequences for HMM training. No public dataset provides this. Original data engineering.

4. **Vectorless RAG for Moderation** — applying PageIndex (Sep 2025) to content moderation is a novel application. No published paper does this.

5. **Session Manipulation Test** — the adversarial test where a user sends warm-up messages before attacking is a novel evaluation methodology.

6. **Hinglish/Indian Platform Evaluation** — evaluating on Indian code-switching context is an original contribution relevant to deployment in India.

7. **Ablation Studies** — the 10-row ablation table is original experimental methodology proving each component's contribution.

8. **Custom Intent Taxonomy** — 20-label intent taxonomy (benign / suspicious / malicious tiers) designed specifically for user-to-user chat moderation. No existing dataset or taxonomy covers this exact intent space. The taxonomy definition, annotation protocol, and session-level labeling methodology are original contributions.

### What Would Actually Be Unoriginal

- Download toxic-bert → wrap in Flask → call it a project ❌
- Call GPT-4 API → return response → no ML pipeline ❌

### Academic Precedent

GPT-4 uses pretrained weights from GPT-3. BERT-large uses pretrained weights. Every state-of-the-art model in NLP is built on prior work. The field calls this "transfer learning" and it has been standard methodology since 2018 (Devlin et al.). Using pretrained weights on insufficient training data would be academically wrong.

---

## 16. FYP Report Chapter Structure

### Project Phases — Scope Management

This is a full-scope personal project with no hard deadline. Build in phases to ensure a working end-to-end system at every checkpoint.

```
Phase 1 — MVP (must work end-to-end first)
────────────────────────────────────────────
  ✅ ModernBERT fine-tuning (multi-task: toxicity + intent)
  ✅ HMM session model (Baum-Welch training + Viterbi inference)
  ✅ Trie pre-filter (keyword/slur exact match)
  ✅ Redis session store (sliding window of intent labels)
  ✅ PostgreSQL + pgvector (message log + embedding similarity)
  ✅ FastAPI backend + WebSocket endpoint
  ✅ Streamlit dashboard (fast, ~100 lines — proof of concept)
  ✅ Core test suite (unit + integration + model eval)
  ✅ Ablation studies (10 variants)

Phase 2 — Full Scope (add after Phase 1 is stable)
────────────────────────────────────────────────────
  ⬜ React dashboard (full live moderation UI with Recharts)
  ⬜ TimescaleDB + Grafana (time-series analytics)
  ⬜ Groq API escalation path + PageIndex integration
  ⬜ ONNX export (2–3× inference speedup)
  ⬜ Docker Compose full stack
  ⬜ CI/CD pipeline (GitHub Actions)
  ⬜ Adversarial test suite + fairness evaluation
  ⬜ Load testing (Locust: 100 → 1000 concurrent users)

Future / Stretch (post-project or paper extension)
────────────────────────────────────────────────────
  ⬜ XLM-RoBERTa multilingual extension (Hindi/Hinglish)
  ⬜ Qdrant (if pgvector hits performance limits at scale)
  ⬜ Federated learning (privacy-preserving fine-tuning)
  ⬜ Local Ollama fallback (requires ≥12GB VRAM second machine)
  ⬜ CRF multi-label sequence tagging
```

### Milestone Timeline (Suggested — adjust to your pace)

```
Week  1–3   Dataset prep: Jigsaw download, cleaning, custom intent labeling
Week  4–6   ModernBERT fine-tuning: single-task first, then multi-task
Week  7–8   HMM training: session sequence construction, Baum-Welch
Week  9–10  Pipeline integration: Trie + BERT + HMM + Redis + FastAPI
Week 11–12  Ablation studies: 10 variants, fill results table
Week 13–14  Phase 2: React dashboard, Docker Compose, Groq escalation
Week 15–16  Testing: adversarial, fairness, load testing, Locust
Week 17–18  Report writing: Chapters 1–4
Week 19–20  Report writing: Chapters 5–8 + final review
```

### Recommended Structure (8 Chapters)

```
Chapter 1: Introduction (8–10 pages)
  1.1 Problem Statement
  1.2 Motivation & Industry Context
  1.3 Research Objectives
  1.4 Scope & Limitations
  1.5 Contributions (list your 7 original contributions explicitly)
  1.6 Report Organisation

Chapter 2: Literature Review (15–20 pages)
  2.1 Content Moderation Systems (evolution)
  2.2 Intent Classification in NLP
  2.3 Transformer Models for Text Classification
  2.4 Sequence Modelling (HMM, LSTM, CRF)
  2.5 Retrieval-Augmented Generation
  2.6 Gaps in Existing Work → your contributions fill these

Chapter 3: Theoretical Background (12–15 pages)
  3.1 Transformer Architecture & Attention Mechanism
  3.2 Transfer Learning & Fine-Tuning
  3.3 Hidden Markov Models (Baum-Welch, Viterbi)
  3.4 Loss Functions for Imbalanced Classification
  3.5 Vector Similarity Search (HNSW)
  3.6 Evaluation Metrics (F1, AUC-ROC, Calibration)
  → Use all content from Section 5 of this document

Chapter 4: System Design & Architecture (15–18 pages)
  4.1 Overall System Architecture (the pipeline diagram)
  4.2 Layer-by-Layer Design Decisions
  4.3 Database Architecture (polyglot persistence)
  4.4 API vs Hybrid Decision (Section 9)
  4.5 Real-Time Constraints & Latency Budget
  4.6 PageIndex Integration

Chapter 5: Implementation (20–25 pages)
  5.1 Data Collection & Preprocessing
  5.2 Class Imbalance Handling
  5.3 Model Training Pipeline (ModernBERT fine-tuning)
  5.4 HMM Training (Baum-Welch on session sequences)
  5.5 Decision Engine Implementation
  5.6 PageIndex Knowledge Base Construction
  5.7 Real-Time Pipeline (FastAPI + WebSocket + Redis)
  5.8 Dashboard Implementation

Chapter 6: Experiments & Results (20–25 pages)
  6.1 Experimental Setup (hardware, hyperparameters, splits)
  6.2 Baseline Comparisons
  6.3 Ablation Study Results (10-row table + analysis)
  6.4 Adversarial Test Results (robustness analysis)
  6.5 Latency & Load Test Results
  6.6 Human Evaluation Results
  6.7 Multilingual Evaluation (Hinglish limitation)

Chapter 7: Discussion (10–12 pages)
  7.1 Analysis of Results
  7.2 Comparison with Related Work
  7.3 Limitations
  7.4 Future Work (multilingual, XLM-RoBERTa, federated learning)

Chapter 8: Conclusion (4–5 pages)
  8.1 Summary of Contributions
  8.2 Answers to Research Objectives
  8.3 Final Remarks
```

---

## Quick Reference — Commands

```bash
# Setup
pip install torch transformers peft datasets hmmlearn \
            fastapi uvicorn redis-py asyncpg pgvector \
            scikit-learn imbalanced-learn shap wandb dvc \
            locust pytest pytest-asyncio chromadb faiss-cpu \
            groq openai anthropic                          # escalation API clients

# Train
python train.py --model modernbert --epochs 5 --lr 2e-5 --batch 32

# Evaluate
python evaluate.py --model-path ./checkpoints/best --test-split 0.15

# Ablation
python ablate.py --remove hmm         # no session layer
python ablate.py --model distilbert   # speed variant
python ablate.py --no-finetune        # zero-shot baseline

# Serve
docker-compose up --build

# Load test
locust -f tests/locustfile.py --headless -u 100 -r 10 --run-time 60s

# Export to ONNX
python export_onnx.py --model-path ./checkpoints/best --output ./model-onnx
```

---

*Document generated from complete FYP design session. Last updated: May 2026.*
*For Claude Code: parse each numbered section independently. Section 5 = math context. Section 6 = implementation context. Section 10-12 = test generation context.*
