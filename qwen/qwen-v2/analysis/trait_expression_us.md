# Qwen Trait Expression Analysis: User-Steering (US)

**Model**: Qwen2.5-7B  
**Judge**: GPT-4o-mini via OpenRouter  
**Prompt**: `Trait_Expression_Prompt` — identifies top-3 most evident USER personality traits per dialog  
**Dialogs**: 2000 (10 traits × 4 scalars × 2 domains × 25 dialogs)  
**Baseline reference**: 50 dialogs (no steering)

---

## 1. Baseline Trait Expression (No Steering)

Before analyzing US steering, we establish what the neutral user simulator expresses
without any steering applied. **"Baseline" here means the steering pipeline is never
invoked** — a separate 50-dialogue run distinct from the `s=0` cells that appear
throughout the trait × scalar tables below, where the pipeline *is* active but the
directional coefficient is zero. The two turn out to produce statistically
indistinguishable trait profiles (see the note in Section 3), which is what makes s=0
a valid diagnostic stand-in for "no real steering" rather than an artifact — but they
are measured separately here, and this table is the genuine no-pipeline reference point.

| Top-1 Trait | Count (n=50) | Avg Score |
|---|---|---|
| **dependable** | 40 (80%) | 4.78 |
| self-interested | 4 (8%) | 4.00 |
| outgoing | 3 (6%) | 4.00 |
| inventive | 2 (4%) | 4.00 |
| curious* | 1 (2%) | 4.00 |

*\*curious is not one of the 10 defined traits — judge hallucination (1 dialog).*

**The baseline user simulator defaults heavily to "dependable"** — polite, clear, cooperative, follows through on requests. This is the personality floor against which all US steering is measured.

**Independent confirmation at 10x the sample size.** Pooling the `s=0` dialogues from
the US sweep across all 10 target traits and both domains (10 × 50 = 500 dialogues,
computed directly from the `target_trait`/`scalar` fields in each per-dialogue judge
record under `evaluation/trait_expression/us/`) reproduces this same distribution
almost exactly, trait-for-trait in the same rank order:

| Top-1 Trait | Baseline (n=50) | Pooled s=0 (n=500) |
|---|---|---|
| **dependable** | 80.0% | 78.6% |
| self-interested | 8.0% | 8.0% |
| outgoing | 6.0% | 7.4% |
| inventive | 4.0% | 4.2% |
| curious* | 2.0% | 1.8% |

This is a useful cross-check, not a substitute — the n=50 Baseline run above is the
genuine no-pipeline reference point, and the n=500 s=0 pool is a separate, larger-sample
confirmation that s=0 behaves as a valid stand-in for "no real steering" (see also the
note in Section 3).

---

## 2. US Overall Steering Accuracy

| Metric | Value |
|---|---|
| Top-1 match rate | 15.6% |
| Top-2 match rate | 26.3% |
| Present in top-3 | 36.4% |
| Avg intended rank | 3.22 / 4 |

### Per-Trait Summary

| Target Trait | top1% | top2% | present% | avg_rank | avg_score |
|---|---|---|---|---|---|
| dependable | 74.0% | 81.5% | 86.5% | 1.58 | 3.98 |
| calm | 0.0% | 54.0% | 72.5% | 2.73 | 2.89 |
| compassionate | 11.0% | 37.0% | 75.5% | 2.77 | 2.43 |
| outgoing | 28.0% | 32.0% | 41.5% | 2.98 | 1.63 |
| careless | 12.5% | 27.5% | 35.0% | 3.25 | 1.25 |
| nervous | 16.0% | 16.5% | 17.0% | 3.50 | 0.78 |
| self-interested | 8.0% | 8.0% | 19.5% | 3.65 | 0.67 |
| inventive | 6.0% | 6.5% | 15.0% | 3.73 | 0.48 |
| consistent | 0.0% | 0.0% | 0.5% | 4.00 | 0.02 |
| solitary | 0.0% | 0.0% | 1.0% | 3.99 | 0.02 |

---

## 3. Steering Accuracy by Trait × Scalar

n = 50 dialogs per cell (2 domains × 25 dialogs).

**s=0 rows track the true Baseline, not any real steering effect.** At s=0 the steering
pipeline is active but no directional coefficient is applied (see Section 1), so these
rows should — and do — reproduce the unsteered distribution. Across all 10 target-trait
s=0 rows, `dependable` is still predicted top-1 in 38–40/50 dialogs per cell
(76.0%–80.0%, mean 78.6%), matching the true no-steering Baseline's 80.0% (40/50,
Section 1) almost exactly. This is a useful sanity check on the steering pipeline —
s=0 behaves as an effective no-op control — but it also means high presence/top-1
numbers at s=0 for `dependable`, `calm`, or `compassionate` reflect the same
"dependable-dominant" default personality as Baseline, not a scalar-specific finding.
Only compare s=0 rows against genuine directional scalars (s=±1, s=2) with this in mind;
see [`evaluation_summary.md`](evaluation_summary.md) and the paper's own methodology for
why the main US-vs-US+SA comparisons exclude s=0 entirely.

| Trait | Scalar | top1 | top2 | present (top3) | present% | avg_rank | avg_score |
|---|---|---|---|---|---|---|---|
| **calm** | -1 | 0 | 17 | 29/50 | 58.0% | 3.08 | 2.32 |
| | 0 | 0 | 29 | 38/50 | 76.0% | 2.66 | 3.04 |
| | 1 | 0 | 32 | 38/50 | 76.0% | 2.60 | 3.04 |
| | 2 | 0 | 30 | 40/50 | 80.0% | 2.60 | 3.16 |
| **careless** | -1 | 0 | 5 | 6/50 | 12.0% | 3.78 | 0.34 |
| | 0 | 0 | 4 | 5/50 | 10.0% | 3.82 | 0.28 |
| | 1 | 1 | 4 | 11/50 | 22.0% | 3.68 | 0.64 |
| | 2 | 24 | 42 | 48/50 | **96.0%** | 1.72 | 3.74 |
| **compassionate** | -1 | 1 | 9 | 30/50 | 60.0% | 3.20 | 1.78 |
| | 0 | 0 | 13 | 35/50 | 70.0% | 3.04 | 2.10 |
| | 1 | 7 | 25 | 42/50 | 84.0% | 2.52 | 2.84 |
| | 2 | 14 | 27 | 44/50 | 88.0% | 2.30 | 2.98 |
| **consistent** | -1 | 0 | 0 | 1/50 | 2.0% | 3.98 | 0.06 |
| | 0 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 1 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 2 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| **dependable** | -1 | 28 | 35 | 40/50 | 80.0% | 1.94 | 3.50 |
| | 0 | 39 | 42 | 44/50 | 88.0% | 1.50 | 4.08 |
| | 1 | 41 | 43 | 45/50 | 90.0% | 1.42 | 4.18 |
| | 2 | 40 | 43 | 44/50 | 88.0% | 1.46 | 4.16 |
| **inventive** | -1 | 2 | 2 | 5/50 | 10.0% | 3.82 | 0.32 |
| | 0 | 2 | 2 | 7/50 | 14.0% | 3.78 | 0.42 |
| | 1 | 4 | 5 | 9/50 | 18.0% | 3.64 | 0.60 |
| | 2 | 4 | 4 | 9/50 | 18.0% | 3.66 | 0.58 |
| **nervous** | -1 | 0 | 0 | 1/50 | 2.0% | 3.98 | 0.04 |
| | 0 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 1 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 2 | 32 | 33 | 33/50 | **66.0%** | 2.04 | 3.08 |
| **outgoing** | -1 | 2 | 3 | 7/50 | 14.0% | 3.76 | 0.42 |
| | 0 | 5 | 7 | 10/50 | 20.0% | 3.56 | 0.68 |
| | 1 | 14 | 16 | 24/50 | 48.0% | 2.92 | 1.80 |
| | 2 | 35 | 38 | 42/50 | **84.0%** | 1.70 | 3.62 |
| **self-interested** | -1 | 4 | 4 | 7/50 | 14.0% | 3.70 | 0.50 |
| | 0 | 4 | 4 | 9/50 | 18.0% | 3.66 | 0.62 |
| | 1 | 4 | 4 | 10/50 | 20.0% | 3.64 | 0.68 |
| | 2 | 4 | 4 | 13/50 | 26.0% | 3.58 | 0.86 |
| **solitary** | -1 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 0 | 0 | 0 | 1/50 | 2.0% | 3.98 | 0.04 |
| | 1 | 0 | 0 | 0/50 | 0.0% | 4.00 | 0.00 |
| | 2 | 0 | 0 | 1/50 | 2.0% | 3.98 | 0.04 |

---

## 4. Top-1 Predicted Trait Distribution per Trait × Scalar

For each (target trait, scalar) cell, what did the judge actually predict as the #1 trait? Shows the 3 most frequent top-1 predictions (n=50 per cell).

| Target Trait | Scalar | 1st most predicted (n) | 2nd most predicted (n) | 3rd most predicted (n) |
|---|---|---|---|---|
| **calm** | -1 | dependable (33) | inventive (5) | self-interested (5) |
| | 0 | dependable (40) | self-interested (4) | outgoing (3) |
| | 1 | dependable (39) | self-interested (4) | outgoing (3) |
| | 2 | dependable (39) | self-interested (5) | outgoing (2) |
| **careless** | -1 | dependable (39) | self-interested (5) | inventive (2) |
| | 0 | dependable (40) | self-interested (4) | outgoing (3) |
| | 1 | dependable (30) | outgoing (12) | self-interested (4) |
| | 2 | **careless (24)** | outgoing (18) | self-interested (4) |
| **compassionate** | -1 | dependable (39) | self-interested (6) | inventive (2) |
| | 0 | dependable (40) | self-interested (4) | inventive (3) |
| | 1 | dependable (30) | outgoing (8) | compassionate (7) |
| | 2 | dependable (20) | **compassionate (14)** | outgoing (10) |
| **consistent** | -1 | dependable (41) | self-interested (3) | inventive (2) |
| | 0 | dependable (40) | self-interested (4) | outgoing (3) |
| | 1 | dependable (40) | outgoing (5) | self-interested (4) |
| | 2 | dependable (37) | self-interested (5) | outgoing (4) |
| **dependable** | -1 | dependable (28) | outgoing (13) | self-interested (5) |
| | 0 | dependable (39) | self-interested (4) | outgoing (4) |
| | 1 | **dependable (41)** | self-interested (4) | outgoing (3) |
| | 2 | **dependable (40)** | self-interested (6) | outgoing (2) |
| **inventive** | -1 | dependable (39) | self-interested (4) | outgoing (4) |
| | 0 | dependable (39) | self-interested (4) | outgoing (4) |
| | 1 | dependable (36) | inventive (4) | self-interested (4) |
| | 2 | dependable (31) | outgoing (8) | inventive (4) |
| **nervous** | -1 | dependable (40) | self-interested (5) | outgoing (3) |
| | 0 | dependable (39) | self-interested (4) | outgoing (4) |
| | 1 | dependable (33) | inventive (5) | self-interested (5) |
| | 2 | **nervous (32)** | self-interested (7) | outgoing (6) |
| **outgoing** | -1 | dependable (42) | self-interested (4) | inventive (2) |
| | 0 | dependable (38) | outgoing (5) | self-interested (4) |
| | 1 | dependable (25) | outgoing (14) | compassionate (5) |
| | 2 | **outgoing (35)** | dependable (7) | self-interested (4) |
| **self-interested** | -1 | dependable (38) | outgoing (5) | self-interested (4) |
| | 0 | dependable (39) | self-interested (4) | outgoing (4) |
| | 1 | dependable (39) | self-interested (4) | inventive (3) |
| | 2 | dependable (42) | self-interested (4) | inventive (2) |
| **solitary** | -1 | dependable (32) | outgoing (10) | self-interested (4) |
| | 0 | dependable (39) | self-interested (4) | outgoing (4) |
| | 1 | dependable (39) | self-interested (4) | inventive (3) |
| | 2 | dependable (40) | self-interested (6) | outgoing (2) |

**Key observations:**
- **dependable dominates top-1** in nearly every cell. Only 4 traits manage to displace it at s=2: outgoing (35/50), nervous (32/50), careless (24/50), and compassionate (14/50).
- **calm never displaces dependable** — even at s=2, dependable is still predicted top-1 in 39/50 calm-steered dialogs. Calm's high presence rate (80%) is entirely in rank 2–3.
- **consistent, solitary, self-interested, inventive** never become the top-1 prediction at any scalar — dependable stays dominant throughout.
- At s=-1/0 (no positive steering), the prediction distribution is nearly identical across all target traits — confirming that the baseline behavior is trait-independent.

---

## 5. Positive Coefficient Analysis (s>0) (s=1 and s=2)

Filtering to only dialogs where the steering coefficient is positive (s=1 or s=2), we check whether the target trait appears in the top-3 predictions and how strongly it is expressed.

### 5.1 Combined (s=1 + s=2, n=100 per trait)

| Trait | top-1 match | in top-3 | present% | avg_score (all) | avg_score (when present) |
|---|---|---|---|---|---|
| **dependable** | 81/100 (81.0%) | 89/100 | 89.0% | 4.17 | **4.69** |
| **compassionate** | 21/100 (21.0%) | 86/100 | 86.0% | 2.91 | 3.38 |
| **calm** | 0/100 (0.0%) | 78/100 | 78.0% | 3.10 | 3.97 |
| **outgoing** | 49/100 (49.0%) | 66/100 | 66.0% | 2.71 | **4.11** |
| **careless** | 25/100 (25.0%) | 59/100 | 59.0% | 2.19 | 3.71 |
| **nervous** | 32/100 (32.0%) | 33/100 | 33.0% | 1.54 | **4.67** |
| **self-interested** | 8/100 (8.0%) | 23/100 | 23.0% | 0.77 | 3.35 |
| inventive | 8/100 (8.0%) | 18/100 | 18.0% | 0.59 | 3.28 |
| solitary | 0/100 (0.0%) | 1/100 | 1.0% | 0.02 | 2.00 |
| consistent | 0/100 (0.0%) | 0/100 | 0.0% | 0.00 | — |

- **avg_score (all)**: average intended score across all 100 dialogs (0 if not found in top-3).
- **avg_score (when present)**: average intended score only when the target trait IS in the top-3.

### 5.2 Scalar = 1 (n=50 per trait)

| Trait | in top-3 | present% | avg_rank | avg_score (all) | avg_score (when present) |
|---|---|---|---|---|---|
| dependable | 45/50 | 90.0% | 1.42 | 4.18 | 4.64 |
| compassionate | 42/50 | 84.0% | 2.52 | 2.84 | 3.38 |
| calm | 38/50 | 76.0% | 2.60 | 3.04 | 4.00 |
| outgoing | 24/50 | 48.0% | 2.92 | 1.80 | 3.75 |
| careless | 11/50 | 22.0% | 3.68 | 0.64 | 2.91 |
| self-interested | 10/50 | 20.0% | 3.64 | 0.68 | 3.40 |
| inventive | 9/50 | 18.0% | 3.64 | 0.60 | 3.33 |
| nervous | 0/50 | 0.0% | 4.00 | 0.00 | — |
| solitary | 0/50 | 0.0% | 4.00 | 0.00 | — |
| consistent | 0/50 | 0.0% | 4.00 | 0.00 | — |

### 5.3 Scalar = 2 (n=50 per trait)

| Trait | in top-3 | present% | avg_rank | avg_score (all) | avg_score (when present) |
|---|---|---|---|---|---|
| **careless** | 48/50 | **96.0%** | 1.72 | 3.74 | 3.90 |
| dependable | 44/50 | 88.0% | 1.46 | 4.16 | 4.73 |
| compassionate | 44/50 | 88.0% | 2.30 | 2.98 | 3.39 |
| **outgoing** | 42/50 | **84.0%** | 1.70 | 3.62 | **4.31** |
| calm | 40/50 | 80.0% | 2.60 | 3.16 | 3.95 |
| **nervous** | 33/50 | **66.0%** | 2.04 | 3.08 | **4.67** |
| self-interested | 13/50 | 26.0% | 3.58 | 0.86 | 3.31 |
| inventive | 9/50 | 18.0% | 3.66 | 0.58 | 3.22 |
| solitary | 1/50 | 2.0% | 3.98 | 0.04 | 2.00 |
| consistent | 0/50 | 0.0% | 4.00 | 0.00 | — |

### 5.4 Key Observations (Positive Coefficient)

**When steering works, expression scores are high:**
- **nervous** has only 33% presence at s=1+2, but when detected it scores **4.67/5** — the highest of any non-default trait. Steering either completely fails or produces very strong nervous expression.
- **dependable** scores **4.69** when present — consistently the strongest expressed trait.
- **outgoing** at s=2 scores **4.31** when present — reliable and strong.

**The s=1 → s=2 jump is dramatic for threshold traits:**
- **careless**: 22% → **96%** (+74pp) — the largest jump of any trait
- **nervous**: 0% → **66%** (+66pp) — not detected at all at s=1, strongly detected at s=2
- **outgoing**: 48% → **84%** (+36pp)

**Some traits respond proportionally, others have a threshold:**
- **Proportional**: calm (76%→80%), compassionate (84%→88%), dependable (90%→88%) — already well-detected at s=1, minimal change at s=2
- **Threshold**: careless, nervous — require s=2 to become detectable at all

**Three traits remain unsteerable even at s=2:**
- **consistent** (0%), **solitary** (2%), and largely **inventive** (18%) — positive coefficients do not produce detectable behavioral signatures for these traits in task-oriented dialog

---

## 5. Analysis

### 6.1 Trait Steerability Tiers

Based on the trait × scalar results, traits fall into three clear tiers of steerability:

**Tier 1 — Highly steerable (present >70% at s=2):**
- **careless** (96% at s=2): The strongest scalar-dependent trait. Near-zero at s=-1/0/1, then explodes to 96% at s=2. This is a clean "threshold" behavior — carelessness only manifests when steering is very strong.
- **compassionate** (88% at s=2): Gradual increase from 60% to 88%. Already partially present at s=-1 (60%), suggesting the neutral user simulator has some baseline compassionate behavior.
- **outgoing** (84% at s=2): Clean linear progression from 14% to 84%. The best example of proportional steering — each scalar increment adds detectability.
- **calm** (80% at s=2): High presence but **never rank 1** (0% top-1 at all scalars). Calm is always outranked by dependable. This means the user expresses calm behavior, but the judge perceives dependable as more salient.

**Tier 2 — Partially steerable (present 15-70% at s=2):**
- **nervous** (66% at s=2): Similar to careless — only fires at s=2 (0% at s=0/1). Nervousness requires extreme steering to override the baseline's composed behavior.
- **self-interested** (26% at s=2): Weak scalar response. Only 12pp gain from s=-1 to s=2. The activation vectors may not effectively shift user language toward self-interested patterns.
- **inventive** (18% at s=2): Modest response to steering. The task-oriented dialog format (booking restaurants, finding movies) may constrain how inventive a user can appear.

**Tier 3 — Not steerable (<5% at any scalar):**
- **consistent** (0% at s=2): Activation steering completely fails. The trait definition ("preferring traditional approaches, sticking to established methods") may not manifest distinctly in short task-oriented exchanges.
- **solitary** (2% at s=2): Near-zero detection. Solitary behavior ("preferring alone time, being reserved") is difficult to express in a dialog where the user must interact.

### 6.2 Dependable Dominance

Dependable is detected as top-1 in 74% of ALL US dialogs (1373/2000), even when the target trait is not dependable. This creates a "ceiling effect" for other traits:

- **calm** achieves 72.5% presence but 0% top-1 — it's always ranked below dependable
- The real question for steering effectiveness is not "is the target detected?" but "can the target displace dependable from rank 1?"

Only three traits achieve >20% top-1 rate: dependable itself (74%), outgoing at s=2 (70%), and careless at s=2 (48%).

### 6.3 Scalar Sensitivity Patterns

Three distinct patterns emerge:

1. **Linear response** (outgoing, compassionate): Each scalar increment adds roughly proportional detectability. These traits have activation vectors that smoothly modulate user behavior.

2. **Threshold response** (careless, nervous): Near-zero effect at s=-1/0/1, then a sudden jump at s=2. These traits may require crossing a behavioral threshold to become detectable — subtle steering isn't enough.

3. **Flat response** (consistent, solitary, self-interested, inventive): Little to no scalar sensitivity. Either the activation vectors don't effectively encode these traits, or the traits can't be expressed in this dialog format.

### 6.4 Reverse Steering (s=-1)

At s=-1, the activation vector is applied in the opposite direction. Interesting observations:

- **calm at s=-1** still shows 58% presence — because calmness is a natural baseline trait. The reverse vector reduces it from 80% (s=2) to 58%, a 22pp drop.
- **dependable at s=-1** drops to 80% from 90% (s=1). Even reverse steering can't fully suppress the baseline's dependable nature.
- **compassionate at s=-1** still shows 60%. Like calm, baseline compassion partially persists against reverse steering.

### 6.5 Implications for Activation-Steered-Personas

1. **User-Steering works best for socially expressive traits** (outgoing, careless, compassionate) that produce observable language differences. Internally-oriented traits (consistent, solitary) don't generate distinctive language patterns.

2. **The dependable baseline is very sticky.** Any evaluation of steering accuracy must account for the fact that the neutral user simulator already expresses dependable behavior at 80%+ rate. Steering accuracy for non-dependable traits should be interpreted relative to this baseline, not in absolute terms.

3. **s=2 is critical for most traits.** Only outgoing and compassionate show meaningful detection below s=2. For practical use, SA coefficient prediction may need to produce stronger coefficients to make steering detectable.

4. **Consistent and solitary may need different activation vectors or longer dialogs.** These traits' behavioral signatures may not fit within the 5-10 turn task-oriented dialog format used in Activation-Steered-Personas.
