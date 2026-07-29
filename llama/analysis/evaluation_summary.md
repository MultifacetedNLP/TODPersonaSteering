# Llama Standard Evaluation: US vs US+SA

**Model**: Llama-3.1-8B-Instruct  
**Conditions**: US (User-Steering) vs US+SA (User-Steering + SA)  
**Dialogues**: 2000 per condition (80 run×domain combos × 25 dialogs each)  
**Metrics**: method_accuracy, full_api_accuracy, dialog_success_rate, dialog_completion_rate, inform_accuracy, BLEU

---

## Key Finding: Llama-3.1-8B Struggles with TOD Task Format

All absolute numbers are dramatically lower than Qwen (which achieved ~53% method accuracy and ~21% dialog success). Llama-3.1-8B-Instruct rarely generates valid API calls, especially in Restaurants domain.

---

## Overall (80 run×domain entries each)

| Metric | US | US+SA | Δ (US+SA−US) |
|--------|-----|------|-------------|
| Method Accuracy | 4.66% | 3.93% | −0.73pp |
| Full API Accuracy | 2.64% | 2.96% | +0.32pp |
| Dialog Success | 0.90% | 1.10% | +0.20pp |
| Dialog Completion | 0.00% | 0.00% | 0.00pp |
| Inform Accuracy | 0.03% | 0.41% | +0.38pp |
| BLEU | 12.50% | 12.72% | +0.22pp |

**US+SA is marginally better** on most metrics except method accuracy. Dialog completion is 0% for both — Llama never fully completes a dialog in either condition.

---

## By Domain

| Domain | Metric | US | US+SA | Δ |
|--------|--------|----|------|---|
| Movies | Method Acc | 8.45% | 4.86% | −3.59pp |
| Movies | Full API Acc | 4.80% | 3.58% | −1.22pp |
| Movies | Dialog Succ | 1.70% | 1.00% | −0.70pp |
| Movies | Inform Acc | 0.00% | 0.24% | +0.24pp |
| Movies | BLEU | 11.84% | 11.77% | −0.07pp |
| Restaurants | Method Acc | 0.87% | 2.99% | +2.12pp |
| Restaurants | Full API Acc | 0.49% | 2.34% | +1.85pp |
| Restaurants | Dialog Succ | 0.10% | 1.20% | +1.10pp |
| Restaurants | Inform Acc | 0.06% | 0.58% | +0.52pp |
| Restaurants | BLEU | 13.16% | 13.67% | +0.51pp |

**Domain split is stark**: US wins in Movies; US+SA wins clearly in Restaurants. In Restaurants, US barely generates API calls at all (0.87% method acc vs 2.99% for US+SA). SA appears to help Llama make restaurant reservations more reliably.

---

## By Trait

| Trait | Method (US→US+SA) | Full API (US→US+SA) | Dialog Succ (US→US+SA) |
|-------|------------------|--------------------|-----------------------|
| calm | 5.21% → 3.93% | 3.58% → 3.32% | 1.00% → 1.50% |
| careless | 4.60% → 2.84% | 2.91% → 1.22% | 1.50% → 0.50% |
| compassionate | 5.21% → 5.07% | 2.91% → 3.45% | 0.50% → 1.50% |
| consistent | 4.66% → 4.60% | 2.97% → 4.27% | 0.50% → 1.00% |
| dependable | 4.94% → 5.42% | 2.36% → 4.33% | 1.00% → 1.00% |
| inventive | 3.31% → 3.66% | 2.03% → 2.71% | 0.50% → 1.00% |
| nervous | 3.04% → 3.04% | 1.69% → 2.03% | 1.00% → 0.50% |
| outgoing | 5.00% → 3.32% | 2.70% → 2.98% | 1.00% → 2.00% |
| self-interested | 5.41% → 4.13% | 2.70% → 2.98% | 1.50% → 1.00% |
| solitary | 5.21% → 3.25% | 2.57% → 2.30% | 0.50% → 1.00% |

**US+SA tends to help**: compassionate, consistent, dependable, inventive, outgoing.  
**US tends to be better**: careless, self-interested, solitary (traits with less cooperative user behavior — SA may conflict with these).

---

## By Scalar

| Scalar | Method (US→US+SA) | Full API (US→US+SA) | Dialog Succ (US→US+SA) | Inform (US→US+SA) |
|--------|------------------|--------------------|-----------------------|------------------|
| −1 | 3.43% → 3.82% | 2.24% → 2.57% | 0.40% → 1.20% | 0.00% → 0.35% |
| 0 | 7.97% → 4.20% | 4.05% → 3.82% | 2.00% → 1.60% | 0.00% → 0.72% |
| +1 | 4.60% → 3.47% | 2.51% → 2.20% | 0.60% → 0.80% | 0.00% → 0.35% |
| +2 | 2.63% → 4.22% | 1.76% → 3.25% | 0.60% → 0.80% | 0.12% → 0.23% |

**Scalar 0** (neutral steering): US is much stronger (7.97% vs 4.20% method acc) — SA likely interferes when no trait direction is intended.  
**Scalars −1 and +2** (strong directional): US+SA wins — SA helps push the model when steering is strong.

---

## Largest Individual Run Swings (US vs US+SA)

| Run | Metric | US | US+SA | Δ |
|-----|--------|----|------|---|
| outgoing__s-1 | Dialog Succ | 0.00% | 6.00% | +6.00pp |
| compassionate__s0 | Dialog Succ | 2.00% | 4.00% | +2.00pp |
| consistent__s0 | Dialog Succ | 2.00% | 4.00% | +2.00pp |
| dependable__s2 | Method Acc | 3.79% | 10.02% | +6.23pp |
| compassionate__s2 | Method Acc | 2.44% | 7.58% | +5.14pp |
| outgoing__s1 | Method Acc | 6.49% | 0.00% | −6.49pp |
| careless__s-1 | Method Acc | 5.14% | 1.35% | −3.79pp |
| self-interested__s1 | Method Acc | 8.11% | 3.53% | −4.58pp |

---

## Pairwise Comparison (LLM-Judged)

**Metrics:** Truthfulness, Constraint adherence, User satisfaction (usersat)  
**Method:** Each dialogue from US is paired with the matching US+SA dialogue for the same dialog ID; an LLM judge picks the winner (A = US, B = US+SA) or declares a tie.  
**Coverage:** 6,000 pairwise judgements (2,000 per metric): 10 traits × 4 scalars × 2 domains × ~25 pairs each.

### Key Numbers at a Glance

| Metric | US win% | US+SA win% | tie% | US+SA advantage |
|---|---|---|---|---|
| Truthfulness | 5.9% | **24.6%** | 69.5% | **+18.6 pp** |
| User satisfaction | 6.0% | **38.6%** | 55.4% | **+32.6 pp** |
| Constraint | 8.8% | **25.4%** | 65.8% | **+16.6 pp** |

US+SA wins on **all three metrics** by large margins — the clearest and most consistent finding in the entire Llama evaluation. This contrasts sharply with standard-eval results where differences were tiny (≤0.73 pp) and mixed. The LLM judge sees a fundamentally different picture than slot-accuracy metrics: US+SA produces responses that are perceived as more truthful, more constraint-adherent, and more satisfying across the board.

---

### By Domain

| Metric | Movies Δ (US+SA−US) | Restaurants Δ (US+SA−US) |
|---|---|---|
| Truthfulness | +17.5 pp | **+19.8 pp** |
| User satisfaction | +27.7 pp | **+37.4 pp** |
| Constraint | +12.5 pp | **+19.1 pp** |

Unlike Qwen (where Movies showed larger pairwise gains), Llama's US+SA advantage is larger in **Restaurants** across all three metrics. This aligns with the standard-eval finding that US+SA helps Restaurants more (+2.12 pp method accuracy). In Movies, US actually achieves higher standard-eval method accuracy (8.45% vs 4.86%), yet pairwise judges still prefer US+SA by +17.5 pp on truthfulness — suggesting US+SA generates better-sounding dialogue even when it produces fewer correct API calls.

---

### By Trait

| Trait | Truthfulness Δ | Usersat Δ | Constraint Δ |
|---|---|---|---|
| calm | +16.0 pp | +25.5 pp | +17.6 pp |
| careless | +20.0 pp | +31.5 pp | +8.8 pp |
| compassionate | +15.0 pp | +39.0 pp | +14.8 pp |
| consistent | +22.5 pp | +24.0 pp | **+25.0 pp** |
| dependable | +14.0 pp | +36.0 pp | +17.8 pp |
| inventive | **+24.0 pp** | +29.0 pp | **+23.4 pp** |
| nervous | +20.0 pp | **+50.5 pp** | +19.1 pp |
| outgoing | +16.5 pp | +35.0 pp | +13.2 pp |
| self-interested | +18.5 pp | **+40.5 pp** | +11.0 pp |
| solitary | +20.0 pp | +14.5 pp | +14.7 pp |

**Key observations:**
- **US+SA dominates for every trait on every metric** — there is no trait where US wins, unlike Qwen where `consistent` and `solitary` showed US usersat advantages. This is a striking contrast.
- **`nervous`** has the largest usersat advantage (+50.5 pp). Combined with its above-average truthfulness and constraint gains, nervous is by far the trait that benefits most from SA in Llama.
- **`self-interested`** and **`compassionate`** are the next highest on usersat (+40.5 pp and +39.0 pp). These same traits were among US+SA's strongest in Qwen, suggesting a cross-model pattern: high-agency and empathy-driven personas benefit most from SA steering.
- **`solitary`** shows the smallest usersat gain (+14.5 pp) but still clearly favors US+SA. In Qwen, `solitary` was US+SA's worst usersat trait (−11.5 pp). This reversal may reflect that Llama's baseline `solitary` responses are particularly poor in quality, leaving more room for SA to improve them.
- **`consistent`** and **`inventive`** lead on constraint and truthfulness gains. In standard eval, US+SA was already better for `consistent` on full API accuracy and dialog success.

---

### By Scalar

| Scalar | Truthfulness Δ | Usersat Δ | Constraint Δ |
|---|---|---|---|
| s = −1 | +16.8 pp | +33.0 pp | +17.7 pp |
| s = 0 | **+22.2 pp** | +32.2 pp | **+19.9 pp** |
| s = 1 | +19.8 pp | **+33.4 pp** | +18.3 pp |
| s = 2 | +15.8 pp | +31.6 pp | +10.3 pp |

**Key observations:**
- US+SA leads at **every scalar level** — there is no scalar where US wins, unlike Qwen where US won constraint at scalar=0 and most scalars.
- The advantage is remarkably **flat across scalars** (31–33 pp on usersat, 16–22 pp on truthfulness), suggesting SA consistently improves perceived response quality in Llama regardless of how strongly the persona is steered. This differs from Qwen, where gains scaled with scalar strength.
- The largest truthfulness and constraint gains occur at **scalar=0**, the exact opposite of the Qwen pattern (where scalar=0 was anomalously bad for US+SA constraint). In Llama, even neutral persona steering benefits from SA conditioning.
- Standard eval showed US winning at scalar=0 (7.97% vs 4.20% method accuracy). Pairwise reverses this: US+SA wins the largest constraint advantage (+19.9 pp) and second-largest truthfulness gain (+22.2 pp) at scalar=0. The contradiction reflects that US+SA at scalar=0 may sacrifice API call structure for more natural, constraint-adherent language.

---

### Cross-Signal Alignment

| Trait | Standard verdict | Pairwise truthfulness | Pairwise usersat | Agreement? |
|---|---|---|---|---|
| compassionate | US+SA better | US+SA +15 pp | US+SA +39 pp | ✓ Aligned |
| consistent | US+SA better (task) | US+SA +22.5 pp | US+SA +24 pp | ✓ Aligned |
| dependable | US+SA better | US+SA +14 pp | US+SA +36 pp | ✓ Aligned |
| outgoing | US+SA better | US+SA +16.5 pp | US+SA +35 pp | ✓ Aligned |
| careless | US better | US+SA +20 pp | US+SA +31.5 pp | **Divergence** |
| self-interested | US better | US+SA +18.5 pp | US+SA +40.5 pp | **Divergence** |
| solitary | US better | US+SA +20 pp | US+SA +14.5 pp | **Divergence** |
| nervous | Mixed | US+SA +20 pp | US+SA +50.5 pp | Divergence (pairwise stronger) |

**Notable divergences:**
- `careless`, `self-interested`, and `solitary` are the three traits standard eval identifies as US-better (lower method/API accuracy with US+SA). Yet pairwise judgements give US+SA a large advantage on all three. This suggests that for these traits, US+SA generates fewer valid API calls but produces subjectively better dialogue — quality over correctness.
- `nervous` showed mixed standard-eval results (minimal method accuracy change). Pairwise reveals it has the single largest US+SA advantage of any trait (+50.5 pp usersat), suggesting nervous persona under SA steering produces dramatically more engaging and satisfying responses that standard metrics completely miss.
- The overall picture for Llama is one of near-total divergence between standard and pairwise evaluation: standard eval finds small, mixed US/US+SA differences; pairwise finds large, consistent US+SA superiority.

---

## Summary

1. **Llama-3.1-8B is substantially weaker than Qwen** at TOD task execution — method accuracy is ~4% vs ~53% for Qwen, dialog success is <2% vs ~21%. This is a fundamental model capability gap, not a steering artifact.

2. **Standard metrics vs pairwise judgements diverge completely.** Standard eval finds marginal, mixed differences (≤0.73 pp); pairwise LLM judgements find US+SA wins all three metrics by large margins (+16.6 pp constraint, +18.6 pp truthfulness, +32.6 pp usersat). SA produces markedly better-perceived responses even when it does not improve structural task metrics.

3. **US+SA is better overall** for dialog success (+0.20pp), inform accuracy (+0.38pp), and BLEU (+0.22pp), but US has higher method accuracy (+0.73pp). The differences are small relative to the low absolute performance. Pairwise results make clear that this tradeoff favors US+SA on any quality dimension the LLM judge can perceive.

4. **Domain asymmetry (standard eval):** SA clearly helps in Restaurants (+2.12pp method acc) but hurts in Movies (−3.59pp). The Restaurants domain may benefit more from the structured SA signal. **Pairwise results show US+SA winning in both domains**, with larger gains in Restaurants — consistent with the standard-eval domain pattern.

5. **Scalar 0 paradox (standard eval):** US at scalar=0 achieves the highest method accuracy of any condition (7.97%), suggesting the neutral user-steering prompt alone is sufficient. **Pairwise reverses this:** US+SA wins its largest constraint and truthfulness gains at scalar=0 (+19.9 pp and +22.2 pp). US+SA at scalar=0 trades API call accuracy for better-perceived response quality.

6. **US+SA advantages are uniform across all traits and scalars in pairwise evaluation.** Unlike Qwen (where `consistent` and `solitary` flipped negative on usersat, and scalar=0 was anomalous), Llama shows no trait or scalar where US wins a pairwise metric. The SA benefit is pervasive, suggesting Llama's baseline US responses are consistently weaker in quality and have more room for improvement.

7. **`nervous` is the standout trait**: +50.5 pp pairwise usersat advantage, the largest of any trait×metric combination across both models. Combined with above-average truthfulness (+20 pp) and constraint gains (+19.1 pp), the `nervous` persona under SA steering is the single biggest beneficiary of SA in the Llama experiments.

8. **Dialog completion = 0%** for both conditions across all 2000 dialogs. Llama-3.1-8B does not successfully complete end-to-end task dialogs in either condition. The pairwise gains reflect improvements in individual turn quality, not end-to-end task success.
