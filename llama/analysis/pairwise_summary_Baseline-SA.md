# Llama Pairwise Comparison: Baseline vs SA

**Model**: Llama-3.1-8B-Instruct  
**Judge**: GPT-4o-mini via OpenRouter  
**Conditions**: Baseline (no steering) vs SA (SA system-side only)  
**Pairs**: 50 per metric (2 domains × 25 dialogs)  
**Metrics**: User Satisfaction (usersat) | Constraint Satisfaction (constraint) | Truthfulness

---

## Overall Win Rates

| Metric | Baseline Wins | SA Wins | Ties | n_pairs | SA Advantage |
|--------|--------------|---------|------|---------|-------------|
| User Satisfaction | 6.0% | **28.0%** | 66.0% | 50 | **+22.0 pp** |
| Constraint Satisfaction | 2.9% | **32.4%** | 64.7% | 34* | **+29.4 pp** |
| Truthfulness | 10.0% | **30.0%** | 60.0% | 50 | **+20.0 pp** |

*Constraint: only pairs where the dialog had evaluable slot constraints (16 of 50 pairs had no required slots).

**SA wins across all 3 metrics.** SA system-side activation steering improves dialogue quality even without any user-side steering.

---

## By Domain

### User Satisfaction
| Domain | Baseline | SA | Ties | n | SA Advantage |
|--------|----------|-----|------|---|-------------|
| Movies_1_Movies_3 | 8.0% | 24.0% | 68.0% | 25 | +16.0 pp |
| Restaurants_2 | 4.0% | **32.0%** | 64.0% | 25 | **+28.0 pp** |
| **Average over domains** | **6.0%** | **28.0%** | 66.0% | — | **+22.0 pp** |

### Constraint Satisfaction
| Domain | Baseline | SA | Ties | n | SA Advantage |
|--------|----------|-----|------|---|-------------|
| Movies_1_Movies_3 | 7.7% | **46.2%** | 46.2% | 13 | **+38.5 pp** |
| Restaurants_2 | 0.0% | 23.8% | 76.2% | 21 | +23.8 pp |
| **Average over domains** | **3.8%** | **35.0%** | 61.2% | — | **+31.1 pp** |

> Note: constraint's domain average (3.8%/35.0%) differs from the pooled overall figure (2.9%/32.4%, line 16) because Movies_1_Movies_3 has fewer evaluable pairs (n=13) than Restaurants_2 (n=21) — pooling weights Restaurants_2 more heavily, while the domain average weights both domains equally.

### Truthfulness
| Domain | Baseline | SA | Ties | n | SA Advantage |
|--------|----------|-----|------|---|-------------|
| Movies_1_Movies_3 | 16.0% | 28.0% | 56.0% | 25 | +12.0 pp |
| Restaurants_2 | 4.0% | **32.0%** | 64.0% | 25 | **+28.0 pp** |
| **Average over domains** | **10.0%** | **30.0%** | 60.0% | — | **+20.0 pp** |

**Restaurants_2 shows the largest SA advantage** in user satisfaction (+28 pp) and truthfulness (+28 pp). Movies shows the largest SA advantage in constraint satisfaction (+38.5 pp), though n is small (13 evaluable pairs).

---

## Notes

- No by-trait or by-scalar breakdown: Baseline and SA are single neutral runs with no trait/scalar variation.
- 50 pairs is a much smaller sample than the US vs US+SA comparison (2000 pairs). Interpret per-domain numbers (n=13–25) with caution.
- Constraint n=34 because 16 of 50 dialog pairs had no `user_req_slots` for the judge to evaluate.

---

## Comparison with Qwen Baseline vs SA

| Metric | Llama SA Adv | Qwen SA Adv |
|--------|-------------|-------------|
| User Satisfaction | **+22.0 pp** | +12.0 pp |
| Constraint Satisfaction | **+29.4 pp** | 0.0 pp |
| Truthfulness | **+20.0 pp** | +2.0 pp |

SA brings much larger gains over Baseline for Llama than for Qwen. This suggests Llama's baseline is weaker and benefits more from SA's system-side activation steering, while Qwen's baseline is already stronger and less affected by SA alone.

---

## Key Findings

1. **SA consistently beats Baseline on all three metrics for Llama**, with SA advantages of +22 pp (usersat), +29.4 pp (constraint), and +20 pp (truthfulness).

2. **Restaurants benefits most from SA** in user satisfaction and truthfulness (+28 pp each). SA helps Llama handle multi-step reservation tasks more coherently — consistent with the US vs US+SA finding.

3. **Movies shows the strongest SA constraint advantage** (+38.5 pp, n=13). However, the small sample size means this should be interpreted cautiously.

4. **Llama gains far more from SA than Qwen does.** Qwen's constraint advantage is 0 pp and truthfulness is only +2 pp, while Llama gains +29.4 pp and +20 pp respectively. SA appears to provide a larger floor-lift for the weaker Llama baseline.

5. **High tie rates** (60–67%) reflect Llama's generally low task completion — most dialogs are short and incomplete regardless of condition. The judge rates them as equivalent when neither manages meaningful progress.
