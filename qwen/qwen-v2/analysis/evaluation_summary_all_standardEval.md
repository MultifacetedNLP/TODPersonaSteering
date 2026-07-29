# Standard Evaluation Summary — All Four Conditions

**Model:** Qwen2.5-7B-Instruct  
**Conditions:**
- **Baseline** — no personality (no user steering) and no SA (`qwen-baseline-2026-05-20-09-19`)
- **SA** — SA only, no user steering (`qwen-sa-2026-05-20-15-47`)
- **US** — User Steering only, no SA (`qwen-user-steer-2026-05-25-15-49`)
- **US+SA** — User Steering + SA (`qwen-user-steer-sa-both-2026-05-20-15-47`)

**Coverage:**
- Baseline / SA: 50 dialogues each (25 per domain, single neutral run — no trait/scalar variation)
- US / US+SA: 2,000 dialogues each (10 traits × 4 scalars × 2 domains × 25 dialogues)

**Metrics:** Method Accuracy, Full API Accuracy, Dialog Success Rate, Dialog Completion Rate, Inform Accuracy, BLEU (all non-LLM)

> **Note on comparability:** Baseline and SA numbers come from a single 50-dialogue run without trait/scalar variation. US and US+SA numbers are averaged over 40 run labels (all trait × scalar combinations). Direct numeric comparison should be interpreted carefully — the Baseline and SA figures represent a neutral, un-personalised user, whereas US/US+SA figures average across a wide range of personality profiles.

---

## Key Numbers at a Glance

| Metric | Baseline | SA | US | US+SA |
|---|---|---|---|---|
| Method Accuracy | 50.15% | 52.32% | 52.95% | **53.66%** |
| Full API Accuracy | **33.08%** | **34.17%** | 32.46% | 32.71% |
| Dialog Success Rate | 18.00% | **22.00%** | 20.75% | 21.30% |
| Dialog Completion Rate | 12.00% | 10.00% | **12.45%** | 12.10% |
| Inform Accuracy | 42.27% | **43.43%** | 36.35% | 35.51% |
| BLEU | 34.14% | **34.70%** | 33.05% | 33.19% |

**Bold** = best value in row.

---

## Overall Analysis

### Effect of SA alone (Baseline → SA)
SA raises Dialog Success by **+4.00 pp** (18% → 22%) and Method Accuracy by **+2.17 pp**, while also improving Inform Accuracy (+1.16 pp) and BLEU (+0.56 pp). The only regression is Dialog Completion (−2 pp). SA alone is beneficial: it steers the user simulation toward more goal-directed behaviour without hurting fluency or information delivery.

### Effect of User Steering alone (Baseline → US)
User Steering raises Method Accuracy by **+2.80 pp** and Dialog Success by **+2.75 pp** but substantially reduces Inform Accuracy (−5.92 pp) and BLEU (−1.09 pp). User steering makes the simulated user more active in driving the conversation but at the cost of information density in system responses.

### Effect of combining both (Baseline → US+SA)
US+SA achieves the highest Method Accuracy (+3.51 pp) but carries the full inform-accuracy penalty of user steering (−6.76 pp). Relative to US alone, US+SA adds a further +0.71 pp method accuracy gain while deepening the inform-accuracy gap (−0.84 pp vs US).

### SA's marginal effect within user-steering (US → US+SA)
Adding SA on top of US gives modest task-accuracy gains (+0.71 pp method, +0.25 pp full API, +0.55 pp dialog success) without hurting BLEU. The main cost is a further reduction in Inform Accuracy (−0.84 pp) and Dialog Completion (−0.35 pp).

---

## By Domain

| Metric | Baseline | SA | US | US+SA |
|---|---|---|---|---|
| **Movies_1_Movies_3** |||||
| Method Accuracy | 45.95% | 45.95% | 46.25% | 46.25% |
| Full API Accuracy | 27.03% | 27.03% | 26.04% | 26.17% |
| Dialog Success | 20.00% | 20.00% | 19.80% | 19.80% |
| Dialog Completion | 8.00% | 4.00% | **4.80%** | 3.90% |
| Inform Accuracy | **45.00%** | **45.00%** | 40.06% | 38.94% |
| BLEU | **35.67%** | 35.20% | 33.72% | 33.48% |
| **Restaurants_2** |||||
| Method Accuracy | 54.35% | 58.70% | 59.65% | **61.06%** |
| Full API Accuracy | 39.13% | **41.30%** | 38.89% | 39.25% |
| Dialog Success | 16.00% | **24.00%** | 21.70% | 22.80% |
| Dialog Completion | 16.00% | 16.00% | 20.10% | **20.30%** |
| Inform Accuracy | 39.53% | **41.86%** | 32.63% | 32.07% |
| BLEU | 32.61% | **34.20%** | 32.37% | 32.89% |

### Domain observations

**Movies_1_Movies_3:** All four conditions perform similarly on task metrics. User steering (US/US+SA) consistently reduces Inform Accuracy (−4–6 pp vs Baseline/SA) and BLEU (−2 pp). SA adds no task-accuracy benefit in this domain regardless of whether user steering is present. Movie recommendation is open-ended and vague, leaving little structured benefit for SA to exploit.

**Restaurants_2:** The domain where all interventions show clear benefit. SA alone raises Dialog Success from 16% to 24% (+8 pp), the single largest gain across the entire comparison. US+SA achieves the highest Method Accuracy (61.06%). US/US+SA recover the Dialog Completion rate (20% vs 16% Baseline), which SA alone does not improve. The trade-off: US/US+SA drop Inform Accuracy by ~7–8 pp versus Baseline/SA in this domain.

---

## By Trait (US vs US+SA — deltas)

The trait breakdown is only available for US vs US+SA (Baseline and SA have no trait variation).

| Trait | Method Acc Δ | Full API Δ | Dialog Succ Δ | Dialog Comp Δ | Inform Acc Δ | Verdict |
|---|---|---|---|---|---|---|
| calm | −0.80 pp | −0.82 pp | **−1.50 pp** | **−2.50 pp** | −0.90 pp | **US better** |
| careless | +0.68 pp | +0.07 pp | +0.50 pp | +1.50 pp | **−2.17 pp** | Mixed |
| compassionate | +0.68 pp | +0.48 pp | **+1.50 pp** | +1.00 pp | +0.33 pp | **US+SA better** |
| consistent | **+1.97 pp** | **+1.60 pp** | **+1.50 pp** | +0.50 pp | +0.75 pp | **US+SA better** |
| dependable | +0.27 pp | −0.82 pp | 0.00 pp | **−2.00 pp** | −1.25 pp | **US better** |
| inventive | +1.36 pp | 0.00 pp | 0.00 pp | −0.50 pp | +0.62 pp | US+SA marginal |
| nervous | **+1.56 pp** | +0.68 pp | +1.00 pp | +1.00 pp | **−2.75 pp** | Mixed |
| outgoing | +1.02 pp | −0.07 pp | 0.00 pp | +1.00 pp | **−2.17 pp** | Mixed |
| self-interested | +0.14 pp | **+1.91 pp** | **+2.50 pp** | **−2.00 pp** | +0.32 pp | **US+SA better (task)** |
| solitary | +0.20 pp | −0.54 pp | 0.00 pp | −1.50 pp | −1.19 pp | US marginal |

**US+SA helps most:** `consistent`, `self-interested`, `compassionate` — high-agency, multi-goal traits where SA steering aligns with assertive task pursuit.  
**US better:** `calm`, `dependable` — naturally cooperative traits where SA over-steers and reduces efficiency.  
**Inform accuracy cost:** SA consistently reduces inform accuracy for `nervous` (−2.75 pp), `careless` (−2.17 pp), `outgoing` (−2.17 pp) — traits where SA may prioritise expressiveness over information delivery.

---

## By Scalar (US vs US+SA — deltas)

| Scalar | Method Acc Δ | Full API Δ | Dialog Succ Δ | Dialog Comp Δ | Inform Acc Δ |
|---|---|---|---|---|---|
| s = −1 | +0.66 pp | **+1.17 pp** | **+1.00 pp** | 0.00 pp | −1.23 pp |
| s = 0 | +0.73 pp | −0.69 pp | +0.60 pp | −1.20 pp | **−1.50 pp** |
| s = 1 | +0.05 pp | −0.11 pp | 0.00 pp | −0.60 pp | −1.23 pp |
| s = 2 | **+1.38 pp** | +0.62 pp | +0.60 pp | +0.40 pp | +0.60 pp |

US+SA's task-accuracy gains are largest at **s=−1** and **s=2** (strong personality signals in either direction). At **s=0** (trait at its mean), US+SA adds noise without benefit — the inform-accuracy penalty is widest here. SA is most useful when there is a clear personality direction to amplify.

---

## Largest Individual Run-Level Swings (US vs US+SA)

### US+SA wins (>3 pp on any metric)

| Run | Metric | US | US+SA | Δ |
|---|---|---|---|---|
| `careless__s-1` | Dialog Completion | 8% | **16%** | +8.0 pp |
| `inventive__s2` | Method Accuracy | 54.5% | **61.3%** | +6.8 pp |
| `compassionate__s-1` | Dialog Success | 22% | **28%** | +6.0 pp |
| `self-interested__s-1` | Full API Accuracy | 33.9% | **38.2%** | +4.3 pp |
| `consistent__s-1` | Full API Accuracy | 29.6% | **33.9%** | +4.3 pp |
| `consistent__s1` | Dialog Success | 24% | **28%** | +4.0 pp |
| `self-interested__s1` | Dialog Success | 22% | **26%** | +4.0 pp |
| `self-interested__s-1` | Inform Accuracy | 42.0% | **45.8%** | +3.8 pp |
| `consistent__s2` | Full API Accuracy | 29.8% | **33.3%** | +3.5 pp |

### US wins (>3 pp on any metric)

| Run | Metric | US | US+SA | Δ |
|---|---|---|---|---|
| `dependable__s1` | Dialog Completion | **16%** | 8% | −8.0 pp |
| `dependable__s1` | Dialog Success | **26%** | 20% | −6.0 pp |
| `dependable__s1` | Full API Accuracy | **37.4%** | 32.0% | −5.4 pp |
| `careless__s-1` | Inform Accuracy | **34.9%** | 29.9% | −5.0 pp |
| `calm__s-1` | Dialog Success | **24%** | 20% | −4.0 pp |

**`dependable__s1` is the sharpest US+SA regression** (−6 pp dialog success, −8 pp dialog completion, −5.4 pp full API accuracy). The SA coefficients for this run may push the `dependable` trait at scalar=1 in a direction that disrupts task-focused turn-taking.

---

## Summary: What Each Condition Is Good At

| Condition | Strengths | Weaknesses |
|---|---|---|
| **Baseline** | High inform accuracy (42.3%), solid BLEU (34.1%) | Lowest method accuracy, lowest dialog success |
| **SA** | Best dialog success (22%), best full API accuracy (34.2%), best inform accuracy (43.4%), best BLEU (34.7%) | Lowest dialog completion (10%) |
| **US** | Best dialog completion (12.5%), good method accuracy (53.0%) | Low inform accuracy (36.4%), lower BLEU (33.1%) |
| **US+SA** | Highest method accuracy (53.7%) | Lowest inform accuracy (35.5%), compounded US inform cost |

---

## Interpretation

1. **SA alone is the most balanced condition.** It improves task success over Baseline without degrading information delivery. The only cost is a modest drop in dialog completion, likely because SA-adjusted users take more conversational turns before closing.

2. **User Steering substantially reduces inform accuracy.** The −5.9 pp drop from Baseline to US (and −6.8 pp to US+SA) is the clearest systematic trade-off in the experiment. Personality-driven user simulation makes the system more personality-expressive but less informationally dense.

3. **US+SA achieves the best method accuracy** by combining both mechanisms, but it also carries the full inform-accuracy cost of user steering plus a marginal additional cost from SA. It is the right choice when method accuracy and dialog success are the primary objectives.

4. **Domain matters more than condition for Movies.** All four conditions perform nearly identically on Movies_1_Movies_3 task metrics. Restaurants_2 is where the real differentiation occurs, and SA alone shows the single largest gain (+8 pp dialog success over Baseline).

5. **SA helps high-agency traits, hurts cooperative ones.** `consistent`, `self-interested`, and `compassionate` benefit most from US+SA; `calm` and `dependable` see regressions. This pattern is consistent with SA over-steering traits that are already naturally cooperative.

6. **SA adds value most at extreme scalars (s=−1, s=2).** At s=0 it adds noise. Experiment designs targeting strong personality expression will benefit more from SA than neutral conditions.
