# Standard Evaluation Summary — User-Steering (US) vs User-Steering + SA (US+SA)

**Model:** Qwen2.5-7B-Instruct  
**Conditions:** US (`qwen-user-steer-2026-05-25-15-49`) vs US+SA (`qwen-user-steer-sa-both-2026-05-20-15-47`)  
**Coverage — directional scalars only, s=0 excluded** (matches the paper's Table 2 methodology, "averaged over directional user-steering coefficients only, α ∈ {−1, 1, 2}"): 60 run×domain combinations per condition (10 traits × 3 scalars × 2 domains), 25 dialogues each → 1,500 dialogues per condition. The s=0 sweep (20 more run×domain combos, 500 more dialogues per condition) was generated and is still on disk under `evaluation/{us,us_sa}/*__s0/`, but is excluded from every table below to match the paper.  
**Metrics:** Method Accuracy, Full API Accuracy, Dialog Success Rate, Dialog Completion Rate, Inform Accuracy, BLEU (non-LLM, no judge)

---



## By Domain

| Metric | US — Movies | US+SA — Movies | US — Restaurants | US+SA — Restaurants |
|---|---|---|---|---|
| Method Accuracy | 46.04% | 46.12% | 61.54% | **62.85%** |
| Full API Accuracy | 25.69% | 25.87% | 39.17% | **40.11%** |
| Dialog Success | 19.60% | 19.60% | 23.60% | **24.67%** |
| Dialog Completion | **3.87%** | 4.00% | **21.33%** | 21.07% |
| Inform Accuracy | **38.40%** | 37.23% | 30.67% | 30.59% |
| BLEU | **33.07%** | 32.77% | 32.26% | **32.67%** |

**Key observation:** US+SA gains are still concentrated in **Restaurants_2** (+1.31 pp method accuracy, +1.07 pp dialog success, +0.94 pp full API accuracy) — the same pattern as the all-scalar version, slightly larger in magnitude. **Movies_1_Movies_3 is close to flat** on task-completion metrics but now shows a small US+SA *loss* on inform accuracy (−1.17 pp) that wasn't as visible before. Restaurant booking's cleaner, more structured slot-filling still appears to make SA-adjusted behaviour more on-task than the vaguer, more open-ended movie-recommendation domain.

---

## By Trait

| Trait | Method Acc Δ | Full API Δ | Dialog Succ Δ | Dialog Comp Δ | Inform Acc Δ | Overall verdict |
|---|---|---|---|---|---|---|
| calm | −1.07 pp | −0.72 pp | **−2.00 pp** | **−2.67 pp** | −1.20 pp | **US better** |
| careless | +0.90 pp | +0.81 pp | +1.33 pp | +2.00 pp | **−1.28 pp** | Mixed (US on inform) |
| compassionate | +0.91 pp | +1.00 pp | **+2.00 pp** | +2.00 pp | +1.22 pp | **US+SA better** |
| consistent | **+2.62 pp** | **+3.35 pp** | **+3.33 pp** | +0.67 pp | +2.01 pp | **US+SA better** |
| dependable | −0.36 pp | −1.45 pp | **−1.33 pp** | **−2.00 pp** | −2.05 pp | **US better** |
| inventive | +1.45 pp | 0.00 pp | −0.67 pp | 0.00 pp | +0.83 pp | US+SA marginal |
| nervous | +0.99 pp | +0.54 pp | +0.67 pp | +0.67 pp | **−2.89 pp** | Mixed (US on inform) |
| outgoing | +0.64 pp | −0.45 pp | −1.33 pp | +1.33 pp | **−3.28 pp** | Mixed |
| self-interested | +0.64 pp | **+2.90 pp** | **+3.33 pp** | −1.33 pp | +0.82 pp | **US+SA better (task)** |
| solitary | +0.26 pp | −0.36 pp | 0.00 pp | −1.33 pp | −0.39 pp | US marginal |

### Trait groupings

**US+SA clearly helps** (`consistent`, `self-interested`, `compassionate`): unchanged grouping from the all-scalar version, and the gains are now *larger* with s=0 excluded — `consistent` Full API Accuracy Δ rises from +1.60pp to +3.35pp, Dialog Success from +1.50pp to +3.33pp. These high-agency, multi-goal traits benefit clearly from SA steering toward more purposeful, goal-directed dialogue.

**US clearly better** (`calm`, `dependable`): also unchanged grouping, also larger in magnitude — `calm`'s dialog-completion drop deepens from −2.50pp to −2.67pp, and `dependable` now loses on Dialog Success too (−1.33pp, was 0.00pp). SA appears to overcorrect traits that naturally produce cooperative, efficient dialogue.

**Mixed results** (`careless`, `nervous`, `outgoing`): same three traits, same pattern — US+SA improves method/dialog-completion but consistently hurts inform accuracy, now even more sharply for `outgoing` (−3.28pp, was −2.17pp) and `nervous` (−2.89pp, was −2.75pp). SA may shift generation toward personality expression at the cost of information density.

---

## By Scalar

| Scalar | Method Acc Δ | Full API Δ | Dialog Succ Δ | Dialog Comp Δ | Inform Acc Δ |
|---|---|---|---|---|---|
| s = −1 | +0.66 pp | **+1.17 pp** | **+1.00 pp** | 0.00 pp | −1.23 pp |
| s = 1 | +0.06 pp | −0.11 pp | 0.00 pp | −0.60 pp | −1.23 pp |
| s = 2 | **+1.38 pp** | +0.62 pp | +0.60 pp | +0.40 pp | +0.60 pp |

(These three rows are exactly the paper's Appendix Table 4, minus its s=0 row — verified to reproduce it digit-for-digit. The s=0 row itself, for reference: Method +0.73pp, Full API −0.69pp, Dialog Succ +0.60pp, Completion −1.20pp, Inform **−1.50pp** — s=0 wasn't a standard-metrics outlier the way it was for pairwise constraint, so dropping it barely moves the overall numbers above.)

**Key observation:** US+SA shows its biggest task-accuracy gain at **scalar=2** (strongest personality steering, +1.38pp method accuracy) and its biggest full-API/dialog-success gain at **scalar=−1** (inverted traits, +1.17pp / +1.00pp). Inform accuracy is US+SA-negative at every directional scalar (−1.23pp at both s=−1 and s=1, +0.60pp only turns positive at s=2) — unlike the pairwise-constraint case, there's no sharp s=0-specific anomaly being masked here; the standard-metric story is qualitatively the same with or without s=0.

---


## Pairwise Comparison (LLM-Judged)

**Metrics:** Truthfulness, Constraint Satisfaction, User satisfaction (usersat)  
**Method:** Each dialogue from US is paired with the matching US+SA dialogue for the same dialog ID; an LLM judge picks the winner (A = US, B = US+SA) or declares a tie.  
**Coverage — directional scalars only, s=0 excluded** (matches the paper's Table 2 methodology, "averaged over directional user-steering coefficients only, α ∈ {−1, 1, 2}"): 10 traits × 3 scalars × 2 domains × 25 pairs = 1,500 pairwise judgements per metric for truthfulness/usersat. Constraint uses the same 1,500-pair denominator but 479 of those pairs have no checkable slot (N/A, counted as a non-win for both sides, per the paper); among the 1,021 decidable pairs the US+SA advantage is +0.88 pp instead of +0.6 pp.

### Key Numbers at a Glance

| Metric | US win% | US+SA win% | tie%* | US+SA advantage |
|---|---|---|---|---|
| Truthfulness | 4.8% | **13.1%** | 82.1% | **+8.3 pp** |
| User satisfaction | 11.4% | **14.1%** | 74.5% | **+2.7 pp** |
| Constraint | 4.5% | **5.1%** | 58.4%* | **+0.6 pp** |

*Constraint tie% is of the full 1,500-pair denominator; the remaining 31.9% are N/A (no checkable slot) pairs, counted as a non-win for both sides rather than folded into "tie." Restricting to the 1,021 decidable pairs: US 6.7% / US+SA 7.5% / tie 85.8%, Δ+0.88 pp.

US+SA wins clearly on **truthfulness** (+8.3 pp) and moderately on **user satisfaction** (+2.7 pp), matching the paper's Table 2. **Constraint adherence** now also favors US+SA (+0.6 pp, +0.88 pp among decidable pairs) — this flips from the previous US-favoring reading once s=0 is excluded as it is considered as baseline.

---

### By Domain

| Metric | Movies Δ (US+SA−US) | Restaurants Δ (US+SA−US) |
|---|---|---|
| Truthfulness | **+8.9 pp** | +7.7 pp |
| User satisfaction | **+3.7 pp** | +1.7 pp |
| Constraint (denom=750/domain) | −0.1 pp | **+1.3 pp** |

Truthfulness and usersat gains are still larger in **Movies** than Restaurants — same direction as before, magnitudes essentially unchanged by dropping s=0. Constraint now tilts slightly US+SA-favoring in Restaurants (+1.3 pp) and is roughly flat in Movies (−0.1 pp), rather than US-favoring in both domains.

---

### By Trait (excluding s=0)

| Trait | Truthfulness Δ | Usersat Δ | Constraint Δ |
|---|---|---|---|
| calm | +7.3 pp | +4.0 pp | 0.0 pp |
| careless | +8.7 pp | +6.7 pp | −1.0 pp |
| compassionate | **+11.3 pp** | +10.7 pp | +2.0 pp |
| consistent | +5.3 pp | **−7.3 pp** | **+7.9 pp** |
| dependable | **+10.0 pp** | −1.3 pp | 0.0 pp |
| inventive | +4.0 pp | +2.0 pp | 0.0 pp |
| nervous | **+12.0 pp** | **+12.0 pp** | −1.0 pp |
| outgoing | +9.3 pp | +2.7 pp | −2.0 pp |
| self-interested | +6.0 pp | **+10.7 pp** | +4.9 pp |
| solitary | +9.3 pp | **−12.7 pp** | −1.9 pp |

**Trait groupings (pairwise, s=0 excluded):**

- **US+SA clearly better on truthfulness and usersat** (`nervous`, `compassionate`, `self-interested`): unchanged from the all-scalar reading — `nervous` still leads both truthfulness (+12.0 pp) and usersat (+12.0 pp); `compassionate` and `self-interested` both stay above +10 pp usersat.

- **US+SA helps truthfulness but hurts usersat** (`consistent`, `solitary`, and now also `dependable`): `consistent` (usersat −7.3 pp) and `solitary` (usersat −12.7 pp, still the largest loss of any trait) hold their direction from the all-scalar reading. **`dependable` flips sign** here — its all-scalar usersat delta was +2.5 pp (US+SA favored), but with s=0 excluded it's −1.3 pp (US favored). The s=0 bucket was doing meaningful work propping up `dependable`'s US+SA usersat number.

- **Constraint is now more clearly trait-dependent, not just uniformly US-favoring**: `consistent` (+7.9 pp) and `self-interested` (+4.9 pp) show the strongest US+SA constraint gains — larger than in the all-scalar reading. `compassionate` turns modestly positive (+2.0 pp, was 0.0). `outgoing` (−2.0 pp), `careless`/`nervous` (−1.0 pp each), and `solitary` (−1.9 pp) remain US-favoring but by smaller margins than before.

---

### By Scalar

| Scalar | Truthfulness Δ | Usersat Δ | Constraint Δ |
|---|---|---|---|
| s = −1 | +7.6 pp | +1.4 pp | +0.3 pp |
| s = 1 | +7.0 pp | +4.0 pp | +0.3 pp |
| s = 2 | **+10.4 pp** | +2.8 pp | +2.0 pp |



**Denominator note:** this table's constraint column uses a *per-scalar evaluable-pairs-only* denominator (matching the paper's Appendix Table 4 methodology exactly, verified to reproduce it: −6.23, +0.30, +0.29, +2.04 pp for s=0,−1,1,2 respectively) — a different convention from the Key Numbers table above, which uses the *full 1,500-pair denominator with N/A pairs counted as a non-win* (matching the paper's main Table 2). The two conventions give different overall constraint figures: averaging this table's three directional rows gives **+0.88 pp** (which is exactly the paper's "among decidable pairs" footnote figure), not the Key Numbers table's **+0.6 pp** (full-denominator figure). Both are correct — they're just answering slightly different questions (see the Key Numbers footnote above for the same distinction). Don't average across the two tables expecting them to reconcile to a single number.)

**Key observations (directional scalars only):**
- Truthfulness advantage still **grows with scalar strength**, peaking at s=2 (+10.4 pp) — unchanged, since none of the three directional rows involved s=0 to begin with.
- **Constraint is now positive at every directional scalar** (+0.30, +0.29, +2.04 pp) — the negative-leaning overall constraint reading in the old all-scalar table was driven heavily by the s=0 anomaly, not by any directional steering level.
- Usersat is fairly flat across directional scalars (+1.4 to +4.0 pp), without the s=0 peak (+5.4 pp) that previously looked like "moderate steering is best" — that read no longer holds once s=0 is excluded.

---



