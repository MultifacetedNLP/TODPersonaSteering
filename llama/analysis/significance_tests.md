# Llama: Significance Tests — Baseline vs SA, and US vs US+SA

## Methodology

**Pairwise judge metrics (constraint, truthfulness, usersat)**: McNemar's exact test on the paired win/loss counts per dialog pair (ties are concordant and excluded from the test, per standard McNemar's methodology). Exact binomial test on `min(wins_a, wins_b)` out of `wins_a + wins_b` trials vs p=0.5.

**Standard automatic metrics (method_accuracy, full_api_accuracy, successful_dialogs_rate, dialog_completion_rate, inform_accuracy, bleu)**: see per-comparison notes below — the two comparisons have very different sample sizes available, so different tests apply.

Significance codes: `*** p<0.001`, `** p<0.01`, `* p<0.05`, `ns` = not significant.

---

## 1. Pairwise judge metrics — McNemar's exact test

### Baseline vs SA (single run per domain, all pairs)

| Metric | n_total | Baseline wins | SA wins | Ties | Discordant | p-value | Sig |
|---|---|---|---|---|---|---|---|
| Constraint | 34 | 1 | 11 | 22 | 12 | 0.00635 | ** |
| Truthfulness | 50 | 5 | 15 | 30 | 20 | 0.0414 | * |
| User Satisfaction | 50 | 3 | 14 | 33 | 17 | 0.0127 | * |

**SA significantly beats Baseline on all three judge metrics**, despite the small sample (n=34–50 pairs). Constraint has the strongest effect.

### US vs US+SA (s=−1, +1, +2 only; s=0 excluded)

| Metric | n_total | US wins | US+SA wins | Ties | Discordant | p-value | Sig |
|---|---|---|---|---|---|---|---|
| Constraint | 1018 (482 skipped, no evaluable slots) | 95 | 252 | 671 | 347 | 1.53e-17 | *** |
| Truthfulness | 1500 | 82 | 344 | 1074 | 426 | 3.08e-39 | *** |
| User Satisfaction | 1500 | 90 | 580 | 830 | 670 | 1.38e-88 | *** |

**US+SA beats US with overwhelming statistical significance on all three judge metrics** (p << 0.001). Usersat has the largest, most decisive effect (670 discordant pairs, 580-vs-90 split).

---

## 2. Standard automatic metrics

### US vs US+SA — Wilcoxon signed-rank test

Paired unit: one (trait × scalar) run-group's domain-level summary value, s=−1/+1/+2 only → **n=60 paired observations** per metric (10 traits × 3 scalars × 2 domains).

| Metric | n | n with nonzero diff | US mean | US+SA mean | Δ (US+SA−US) | p-value | Sig |
|---|---|---|---|---|---|---|---|
| method_accuracy | 60 | 40 | 3.55% | 3.84% | +0.28pp | 0.590 | ns |
| full_api_accuracy | 60 | 37 | 2.17% | 2.67% | +0.50pp | 0.417 | ns |
| successful_dialogs_rate | 60 | 15 | 0.53% | 0.93% | +0.40pp | 0.157 | ns |
| dialog_completion_rate | 60 | 0 | 0.00% | 0.00% | 0.00pp | — | all zero, undefined |
| inform_accuracy | 60 | 6 | 0.04% | 0.31% | +0.27pp | 0.0196 | * |
| bleu | 60 | 60 | 12.61% | 12.42% | −0.19pp | 0.335 | ns |

**Only inform_accuracy shows a significant difference** (US+SA higher). The other task-completion metrics (method_accuracy, full_api_accuracy, dialog_success) trend slightly in US+SA's favor but are **not statistically significant** — consistent with the earlier finding that the LLM judge favors US+SA far more decisively than the standard task metrics do.

### Baseline vs SA

Only **n=2 domain-level observations** exist per condition (Movies_1_Movies_3, Restaurants_2) — there is no by-trait/by-scalar sweep for Baseline/SA, since these are single neutral runs. A paired test with n=2 can never reach significance (minimum two-sided Wilcoxon p=0.5 at n=2), so no such test is reported.

For the two metrics that are true per-dialog binary rates with a **known denominator (25 dialogs/domain)**, Fisher's exact test was run instead on the 2×2 success/failure table:

| Metric | Domain | Baseline | SA | p-value | Sig |
|---|---|---|---|---|---|
| successful_dialogs_rate | Movies_1_Movies_3 | 1/25 | 0/25 | 1.000 | ns |
| successful_dialogs_rate | Restaurants_2 | 0/25 | 0/25 | — | undefined (no successes either side) |
| successful_dialogs_rate | Pooled | 1/50 | 0/50 | 1.000 | ns |
| dialog_completion_rate | both domains | 0/25 | 0/25 | — | undefined (no successes either side) |

No significant difference — task success is too rare in both conditions (1 success out of 50 dialogs total) for any test to detect a difference at this sample size.

**method_accuracy, full_api_accuracy, inform_accuracy**: these are computed per API-call opportunity, not per dialog, and only the aggregate rate is stored in `metrics.json` — the actual call-count denominator isn't recoverable without re-scoring the raw dialogs, so no valid proportion test could be run.

**bleu**: continuous per-dialog score; only the domain-level mean is stored, not the per-dialog distribution, so no valid paired test could be run.

---

## Key takeaways

1. **The LLM-judge quality metrics (constraint/truthfulness/usersat) show statistically robust SA/SA advantages in both comparisons** — significant for Baseline vs SA (n=34–50, p<0.05) and overwhelmingly significant for US vs US+SA (n=1018–1500, p<1e-17).
2. **The standard task-completion metrics tell a different story**: for US vs US+SA, none of method_accuracy, full_api_accuracy, or dialog_success reach significance (only inform_accuracy does, p=0.02) — the apparent gains seen in the raw percentages are within noise at n=60.
3. **Baseline vs SA standard metrics can't be rigorously tested** given the tiny available sample (2 domain aggregates, and near-zero task success rates) — this is a data-availability limitation, not a null finding.
4. Overall: **the case for SA/SA improving dialogue *quality* (per the LLM judge) is statistically solid; the case for it improving *task-completion* metrics is not**, at least not detectably at current sample sizes.
