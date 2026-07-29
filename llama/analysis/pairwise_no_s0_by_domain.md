# Llama Pairwise: US vs US+SA, s=0 excluded, by domain

**Conditions**: US (user-steer, condition_a) vs US+SA (user-steer+sa, condition_b)
**Scalars included**: −1, +1, +2 (s=0 excluded)
**Domains**: Movies_1_Movies_3, Restaurants_2
**Source**: `llama/pairwise_comparison/{constraint,truthfulness,usersat}/<trait>__s{-1,1,2}/<domain>/*.json`, aggregated over all 10 traits.

Note: for **constraint**, pairs with no evaluable slot constraints (`winner: null`) are excluded from n and win-rate denominators, matching the methodology in `pairwise_summary_US-US+SA.md`.

---

## Constraint Satisfaction

| Domain | n | US | US+SA | Tie | Δ (US+SA−US) |
|---|---|---|---|---|---|
| Movies_1_Movies_3 | 389 | 13.9% | 24.4% | 61.7% | +10.5pp |
| Restaurants_2 | 629 | 6.5% | 25.0% | 68.5% | +18.4pp |
| **Average over domains** | — | **10.2%** | **24.7%** | 65.1% | **+14.5pp** |
| Pooled (all pairs) | 1018 | 9.3% | 24.8% | 65.9% | +15.4pp |

## Truthfulness

| Domain | n | US | US+SA | Tie | Δ (US+SA−US) |
|---|---|---|---|---|---|
| Movies_1_Movies_3 | 750 | 4.1% | 17.9% | 78.0% | +13.7pp |
| Restaurants_2 | 750 | 6.8% | 28.0% | 65.2% | +21.2pp |
| **Average over domains** | — | **5.5%** | **22.9%** | 71.6% | **+17.5pp** |
| Pooled (all pairs) | 1500 | 5.5% | 22.9% | 71.6% | +17.5pp |

## User Satisfaction

| Domain | n | US | US+SA | Tie | Δ (US+SA−US) |
|---|---|---|---|---|---|
| Movies_1_Movies_3 | 750 | 5.5% | 31.2% | 63.3% | +25.7pp |
| Restaurants_2 | 750 | 6.5% | 46.1% | 47.3% | +39.6pp |
| **Average over domains** | — | **6.0%** | **38.7%** | 55.3% | **+32.7pp** |
| Pooled (all pairs) | 1500 | 6.0% | 38.7% | 55.3% | +32.7pp |

---

## Takeaways

- US+SA wins across all three metrics in both domains once s=0 is dropped; the gap is never reversed.
- Restaurants_2 shows a consistently larger US+SA advantage than Movies_1_Movies_3 on all three metrics (+18.4 vs +10.5pp constraint, +21.2 vs +13.7pp truthfulness, +39.6 vs +25.7pp usersat) — same pattern as the full (s-included) analysis in `pairwise_summary_US-US+SA.md`.
- Usersat shows the largest domain-averaged US+SA advantage (+32.7pp), followed by truthfulness (+17.5pp) and constraint (+14.5pp).
- For truthfulness/usersat, domain-averaged and pooled numbers coincide (750 pairs in each domain). For constraint they diverge slightly (10.2% vs 9.3% for US) because Movies_1_Movies_3 has far more no-constraint pairs skipped (361/750) than Restaurants_2 (121/750), so pooling weights Restaurants_2 more heavily than a per-domain average does.
