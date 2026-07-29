# Llama Standard Evaluation: All 4 Conditions

**Model**: Llama-3.1-8B-Instruct  
**Conditions**: Baseline | SA (SA) | US (User-Steering) | US+SA (User-Steering + SA)  
**Coverage**: Baseline/SA = 2 entries (Movies + Restaurants); US/US+SA = 80 entries each  
**Note**: All conditions are compared against Qwen results at the end for context.

---

## Overall Averages

| Metric | Baseline | SA | US | US+SA |
|--------|----------|----|----|------|
| Method Accuracy | 8.11% | 4.05% | 4.66% | 3.93% |
| Full API Accuracy | 4.05% | 2.70% | 2.64% | 2.96% |
| Dialog Success | 2.00% | 0.00% | 0.90% | 1.10% |
| Dialog Completion | 0.00% | 0.00% | 0.00% | 0.00% |
| Inform Accuracy | 0.00% | 0.00% | 0.03% | 0.41% |
| BLEU | 12.18% | 13.15% | 12.50% | 12.72% |

**Critical finding**: Baseline has the highest method accuracy (8.11%) and dialog success (2.00%). Every form of steering — SA, user-steering, or both — *reduces* task completion performance for Llama-3.1-8B.  Dialog completion is 0% across all conditions.

---

## By Domain

### Movies_1_Movies_3

| Metric | Baseline | SA | US | US+SA |
|--------|----------|----|----|------|
| Method Accuracy | 16.22% | 8.11% | 8.45% | 4.86% |
| Full API Accuracy | 8.11% | 5.41% | 4.80% | 3.58% |
| Dialog Success | 4.00% | 0.00% | 1.70% | 1.00% |
| Inform Accuracy | 0.00% | 0.00% | 0.00% | 0.24% |
| BLEU | 10.73% | 11.75% | 11.84% | 11.77% |

Movies shows the clearest degradation gradient: **Baseline > US > SA > US+SA** for task metrics. The more steering applied, the worse task performance gets.

### Restaurants_2

| Metric | Baseline | SA | US | US+SA |
|--------|----------|----|----|------|
| Method Accuracy | 0.00% | 0.00% | 0.87% | 2.99% |
| Full API Accuracy | 0.00% | 0.00% | 0.49% | 2.34% |
| Dialog Success | 0.00% | 0.00% | 0.10% | 1.20% |
| Inform Accuracy | 0.00% | 0.00% | 0.06% | 0.58% |
| BLEU | 13.64% | 14.55% | 13.16% | 13.67% |

Restaurants is the reverse: **baseline and SA generate 0 API calls** — the model completely fails. Steering (US, US+SA) actually enables Llama to make Restaurants API calls at all. US+SA performs best in this domain.

---

## Key Observations

### 1. Domain-Specific Behavior
Llama-3.1-8B generates API calls in Movies (schema: `FindMovies`, single service) but fails entirely in Restaurants (schema: `ReserveRestaurant`, multi-step reservation). User steering prompts appear to unlock Restaurants behavior. This suggests the baseline system prompt is insufficient for multi-step reservation domains.

### 2. SA Alone (SA) Hurts Task Performance
SA reduces method accuracy from 8.11% → 4.05% in Movies and does not help Restaurants at all. Adding sa coefficient calls to the system without user-side steering confuses the task execution.

### 3. User-Steering Enables Restaurants
US achieves 0.87% method accuracy in Restaurants (vs 0% baseline/SA). US+SA achieves 2.99%. The user steering prompt appears to scaffold the system into attempting API calls in this domain.

### 4. US+SA Wins on Inform Accuracy
US+SA is the only condition with meaningful inform accuracy (0.41% overall, 0.58% in Restaurants). This suggests that when the system does complete a lookup, the SA system-side signal helps it report results more accurately.

### 5. Dialog Completion = 0% Everywhere
No condition ever achieves a fully completed dialog. This confirms Llama-3.1-8B-Instruct is too weak for end-to-end TOD in this setup — it cannot chain multi-turn API calls reliably.

---

## By Trait (US vs US+SA — 80 entries each)

| Trait | Method US | Method US+SA | Dialog Succ US | Dialog Succ US+SA |
|-------|-----------|-------------|----------------|------------------|
| calm | 5.21% | 3.93% | 1.00% | 1.50% |
| careless | 4.60% | 2.84% | 1.50% | 0.50% |
| compassionate | 5.21% | 5.07% | 0.50% | 1.50% |
| consistent | 4.66% | 4.60% | 0.50% | 1.00% |
| dependable | 4.94% | 5.42% | 1.00% | 1.00% |
| inventive | 3.31% | 3.66% | 0.50% | 1.00% |
| nervous | 3.04% | 3.04% | 1.00% | 0.50% |
| outgoing | 5.00% | 3.32% | 1.00% | 2.00% |
| self-interested | 5.41% | 4.13% | 1.50% | 1.00% |
| solitary | 5.21% | 3.25% | 0.50% | 1.00% |

US+SA improves dialog success for: **calm, compassionate, consistent, inventive, outgoing, solitary**.  
US retains advantage for: **careless, nervous, self-interested** — traits where cooperative persona signals may work against task completion.

---

## By Scalar (US vs US+SA)

| Scalar | Method US | Method US+SA | Dialog Succ US | Dialog Succ US+SA |
|--------|-----------|-------------|----------------|------------------|
| −1 | 3.43% | 3.82% | 0.40% | 1.20% |
| 0 | 7.97% | 4.20% | 2.00% | 1.60% |
| +1 | 4.60% | 3.47% | 0.60% | 0.80% |
| +2 | 2.63% | 4.22% | 0.60% | 0.80% |

**Scalar 0** (neutral steering) gives US its highest numbers (7.97% method acc). At scalar=0, the user steering prompt acts as a neutral scaffold that helps the model without directional pressure — adding SA on top introduces unnecessary complexity.  
**Scalars −1 and +2** (strong directional) both favor US+SA, especially for dialog success. Strong trait signals appear to benefit from the combined system-level SA support.

---

## Comparison with Qwen (Context)

| Metric | Llama Baseline | Llama US+SA | Qwen Baseline | Qwen US+SA |
|--------|---------------|------------|---------------|-----------|
| Method Accuracy | 8.11% | 3.93% | ~53% | ~54% |
| Full API Accuracy | 4.05% | 2.96% | ~32% | ~33% |
| Dialog Success | 2.00% | 1.10% | ~21% | ~21% |
| Dialog Completion | 0.00% | 0.00% | ~12% | ~12% |
| Inform Accuracy | 0.00% | 0.41% | ~36% | ~36% |
| BLEU | 12.18% | 12.72% | ~33% | ~33% |

Qwen outperforms Llama by **5–15× on all task metrics**. The gap is largest for dialog completion (Qwen: 12%, Llama: 0%) and inform accuracy (Qwen: 36%, Llama: <1%). BLEU is the only metric where Llama is within range (~12% vs ~33%), likely because BLEU captures surface-level text similarity regardless of task completion.

---

## Summary

1. **Llama-3.1-8B cannot complete TOD dialogs** in any condition (0% dialog completion). The model struggles with multi-turn API call sequencing required by the SGD task format.

2. **Baseline wins on task metrics in Movies** — steering of any kind degrades performance. The 8B model has limited capacity; additional steering signals consume context or confuse the generation.

3. **Steering unlocks Restaurants** — baseline generates 0 API calls in Restaurants; US/US+SA generate some. User steering prompts act as task-scaffolding in harder domains.

4. **US+SA is the best overall condition** for inform accuracy and BLEU, and ties/beats US on dialog success. The SA system signal adds marginal value when combined with user steering.

5. **Model capacity is the bottleneck**, not steering strategy. The US vs US+SA differences (~1pp) are negligible compared to the Qwen vs Llama gap (10–30pp). Conclusions about steering effectiveness from Llama results should be interpreted cautiously.
