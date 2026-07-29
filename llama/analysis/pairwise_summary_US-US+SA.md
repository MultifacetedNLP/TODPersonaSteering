# Llama Pairwise Comparison: US vs US+SA

**Model**: Llama-3.1-8B-Instruct  
**Judge**: GPT-4o-mini via OpenRouter  
**Conditions**: US (user-steer) vs US+SA (user-steer+sa)  
**Pairs**: 2000 per metric (40 run labels × 2 domains × 25 dialogs)  
**Metrics**: User Satisfaction (usersat) | Constraint Satisfaction (constraint) | Truthfulness

---

## Overall Win Rates

| Metric | US Wins | US+SA Wins | Ties | n_pairs |
|--------|---------|-----------|------|---------|
| User Satisfaction | 6.1% | **38.6%** | 55.4% | 2000 |
| Constraint Satisfaction | 8.8% | **25.4%** | 65.8% | 1359* |
| Truthfulness | 5.9% | **24.6%** | 69.6% | 2000 |

*Constraint: only pairs where at least one dialog had evaluable slot constraints (641 pairs had no constraints to compare).

**US+SA dominates US across all 3 metrics.** Despite US having marginally higher standard task metrics (method accuracy 4.66% vs 3.93%), the LLM judge consistently rates US+SA dialogues as more satisfying, constraint-filling, and truthful.

---

## By Domain

### User Satisfaction
| Domain | US | US+SA | Ties |
|--------|-----|------|------|
| Movies_1_Movies_3 | 5.7% | 33.4% | 60.9% |
| Restaurants_2 | 6.4% | **43.8%** | 49.8% |

### Constraint Satisfaction
| Domain | US | US+SA | Ties |
|--------|-----|------|------|
| Movies_1_Movies_3 | 15.2% | 27.7% | 57.1% |
| Restaurants_2 | 4.9% | **24.0%** | 71.1% |

### Truthfulness
| Domain | US | US+SA | Ties |
|--------|-----|------|------|
| Movies_1_Movies_3 | 4.8% | 22.3% | 72.9% |
| Restaurants_2 | 7.0% | **26.8%** | 66.2% |

**Restaurants_2 shows the largest US+SA advantage** across all metrics. This aligns with standard eval findings where US+SA was the only condition generating meaningful API calls in Restaurants. SA system-side coefficients appear to help Llama engage more coherently with multi-step reservation tasks.

---

## By Trait

### User Satisfaction

| Trait | US | US+SA | Ties | US+SA Advantage |
|-------|-----|------|------|----------------|
| calm | 6.5% | 32.0% | 61.5% | +25.5pp |
| careless | 6.0% | 37.5% | 56.5% | +31.5pp |
| compassionate | 5.0% | 44.0% | 51.0% | +39.0pp |
| consistent | 12.5% | 36.5% | 51.0% | +24.0pp |
| dependable | 4.5% | 40.5% | 55.0% | +36.0pp |
| inventive | 1.0% | 30.0% | 69.0% | +29.0pp |
| **nervous** | 2.0% | **52.5%** | 45.5% | **+50.5pp** |
| outgoing | 7.0% | 42.0% | 51.0% | +35.0pp |
| **self-interested** | 6.5% | **47.0%** | 46.5% | **+40.5pp** |
| solitary | 9.5% | 24.0% | 66.5% | +14.5pp |

Strongest US+SA advantage for **nervous** (+50.5pp) and **self-interested** (+40.5pp). Weakest but still positive for **solitary** (+14.5pp) and **calm** (+25.5pp).

### Constraint Satisfaction

| Trait | US | US+SA | Ties |
|-------|-----|------|------|
| calm | 8.8% | 26.5% | 64.7% |
| careless | 11.8% | 20.6% | 67.6% |
| compassionate | 9.6% | 24.4% | 65.9% |
| consistent | 5.1% | **30.1%** | 64.7% |
| dependable | 9.6% | 27.4% | 63.0% |
| inventive | 6.6% | **29.9%** | 63.5% |
| nervous | 5.9% | 25.0% | 69.1% |
| outgoing | 9.6% | 22.8% | 67.6% |
| self-interested | 11.8% | 22.8% | 65.4% |
| solitary | 9.6% | 24.3% | 66.2% |

Consistent US+SA advantage (~15-25pp) across all traits. Largest for **consistent** and **inventive**.

### Truthfulness

| Trait | US | US+SA | Ties |
|-------|-----|------|------|
| calm | 6.0% | 22.0% | 72.0% |
| careless | 5.5% | 25.5% | 69.0% |
| compassionate | 7.0% | 22.0% | 71.0% |
| consistent | 6.5% | **29.0%** | 64.5% |
| dependable | 7.0% | 21.0% | 72.0% |
| inventive | 4.0% | **28.0%** | 68.0% |
| nervous | 4.5% | 24.5% | 71.0% |
| outgoing | 7.5% | 24.0% | 68.5% |
| self-interested | 6.0% | 24.5% | 69.5% |
| solitary | 5.0% | 25.0% | 70.0% |

Uniform US+SA truthfulness advantage (~15-22pp) across all traits. **consistent** and **inventive** show the strongest signal.

---

## By Scalar

### User Satisfaction

| Scalar | US | US+SA | Ties |
|--------|-----|------|------|
| −1 | 3.8% | 36.8% | 59.4% |
| 0 | 6.2% | 38.4% | 55.4% |
| +1 | 5.2% | 38.6% | 56.2% |
| +2 | 9.0% | **40.6%** | 50.4% |

US+SA advantage is stable across all scalars (~30-37pp). Scalar +2 (maximum trait direction) shows the highest US+SA win rate (40.6%) and lowest tie rate (50.4%).

### Constraint Satisfaction

| Scalar | US | US+SA | Ties | n |
|--------|-----|------|------|---|
| −1 | 8.8% | 26.5% | 64.6% | 339 |
| 0 | 7.3% | 27.3% | 65.4% | 341 |
| +1 | 9.1% | 27.4% | 63.4% | 339 |
| +2 | 10.0% | 20.3% | 69.7% | 340 |

> **Note on n**: n is the number of dialog pairs at each scalar that had evaluable constraints — meaning at least one dialog in the pair had required slots (`user_req_slots`) that the judge could check. Out of 500 total pairs per scalar, only ~339–341 had constraint-relevant slot fills; the rest were skipped. This is why the constraint total (1359) is lower than usersat/truthfulness (2000).

US+SA leads at all scalars. Notably, scalar +2 has the lowest US+SA win rate (20.3%) — strong steering may reduce how often both systems satisfy constraints (more ties and US wins at extreme scalar).

### Truthfulness

| Scalar | US | US+SA | Ties |
|--------|-----|------|------|
| −1 | 5.6% | 22.4% | 72.0% |
| 0 | 7.2% | **29.4%** | 63.4% |
| +1 | 5.8% | 25.6% | 68.6% |
| +2 | 5.0% | 20.8% | 74.2% |

Scalar 0 (neutral) gives the highest US+SA truthfulness advantage (29.4%). Extreme scalars (−1, +2) show more ties and lower win rates for both sides — strong directional steering produces less decisive truthfulness differences.

---

## Key Findings

1. **US+SA is the clear winner across all 3 qualitative metrics**, despite US achieving slightly higher standard task metrics (method accuracy). The LLM judge sees SA system-side signals as strongly beneficial for dialogue quality.

2. **Restaurants amplifies US+SA advantages**: usersat +37pp, constraint +19pp, truthfulness +20pp in Restaurants vs Movies. SA helps Llama handle multi-step reservation scenarios more coherently.

3. **Nervous and self-interested show largest usersat gaps** (50.5pp and 40.5pp). These traits may require strong system-side persona alignment to generate convincing user-facing behaviour — pure user steering produces noticeably weaker dialogues for these.

4. **Solitary is the weakest case for US+SA** (usersat +14.5pp). This may be because a solitary user naturally produces fewer turns and less expressive language, where SA adds less value.

5. **Scalar +2 maximises user satisfaction wins for US+SA** (40.6%) but reduces constraint satisfaction wins (20.3%). At extreme trait strength, SA helps with conversational quality but may trade off against task grounding.

6. **High tie rates** (55-70%) across all metrics reflect Llama's low absolute capability — most dialogues are short, incomplete, and similar regardless of condition. The judge rates them as equivalent when neither manages meaningful task progress.

7. **Contrast with standard metrics**: Standard eval (method accuracy, dialog success) slightly favoured US in Movies. Pairwise judge consistently favours US+SA everywhere. This suggests standard metrics under-credit qualitative dialogue improvements that SA enables — the model completes fewer API calls with US+SA but produces better conversational content.
