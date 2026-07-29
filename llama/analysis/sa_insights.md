# Persona-Flow as a System Intervention — Research Insights

**Experiment:** llama-v2 · **Model:** meta-llama/Llama-3.1-8B-Instruct · **Analysis date:** 2026-05-29

---

## Framing

**Persona-flow (SA)** is the experimental intervention applied to the *system model* — the service agent. It is a prompt-based conditioning strategy that makes the system more adaptive, proactive, and personality-aware.

**User-steer** is the experimental variable used to test robustness: steering vectors are injected into the *user model* to simulate users with different Big-5-derived personality traits (10 traits × 4 scalars: s-1, s0, s1, s2). This creates 40 distinct user behavior profiles.

The central research question is: **Does applying sa to the system model improve task-oriented dialog outcomes, and does the answer depend on the user's personality?**

---

## 1. SA standalone: effect against a standard user

Against a standard (non-steered) user, sa produces the following changes:

| Metric                         |    Baseline  |  Persona-Flow  |      Δ     |
| ------------------------------ | ------------ | -------------- | ---------- |
| Dialog success rate            |         2.0% |           0.0% |    -2.0 pp |
| Method accuracy                |         7.2% |           3.6% |    -3.6 pp |
| Full API accuracy              |         3.6% |           2.4% |    -1.2 pp |
| BLEU                           |        12.1% |          13.0% |    +1.0 pp |
| BERT F1                        |        58.8% |          58.4% |    -0.5 pp |
| Dialog completion              |         0.0% |           0.0% |    +0.0 pp |
| Constraint satisfaction        |         2.7% |          14.7% |   +12.0 pp |
| Inform score (LLM)             |        2.343 |          2.400 |     +0.057 |
| Truthfulness (LLM)             |        4.974 |          5.000 |     +0.026 |
| User satisfaction (LLM)        |        1.691 |          1.770 |     +0.078 |
| Perplexity                     |       19.223 |         22.839 |     +3.616 |

The headline result is the dialog success rate change (drop of 2.0 pp, 100% relative loss).

---

## 2. SA effect across user personalities — 40-condition analysis

Comparing user-steer alone (US) vs user-steer + sa (US+SA) across all 40 conditions:

| Outcome                                           | Count               |
| ------------------------------------------------- | ------------------- |
| SA improves constraint satisfaction (>+1 pp)  | **29** / 40 conditions |
| SA is neutral (±1 pp)                        | **2** / 40 conditions |
| SA worsens constraint satisfaction (>−1 pp)  | **9** / 40 conditions |

The aggregate mean over all 40 conditions:

|                         | US mean       | US+SA mean    | Δ         |
| ----------------------- | ------------- | ------------- | --------- |
| Constraint satisfaction  |          8.2% |         12.8% |  +4.64 pp |
| Dialog success rate      |          1.0% |          1.1% |  +0.05 pp |
| Truthfulness             |         4.080 |         3.890 |     -0.19 |
| User satisfaction        |         1.708 |         1.752 |     +0.04 |

---

## 3. SA effect by user performance tier

Segmenting by baseline constraint satisfaction (without SA):

| User type                           | US csat (mean) | US+SA csat (mean) | SA Δ       |
| ----------------------------------- | -------------- | ----------------- | ---------- |
| Low-performing (csat < 70%)         |           8.2% |             12.8% |   +4.64 pp |
| Mid-performing (csat 70–75%)        |            N/A |               N/A |        N/A |
| High-performing (csat > 75%)        |            N/A |               N/A |        N/A |

## 4. Per-trait SA effect on constraint satisfaction

Mean constraint satisfaction across all scalars for each trait:

| Trait            | US csat | US+SA csat | Δ          |
| ---------------- | ------- | ---------- | ---------- |
| **inventive       ** |    6.0% |      15.3% |   +9.35 pp |
| **consistent      ** |    6.6% |      13.4% |   +6.77 pp |
| **calm            ** |    7.6% |      14.0% |   +6.46 pp |
| **nervous         ** |    7.6% |      13.4% |   +5.79 pp |
| **compassionate   ** |    9.2% |      14.9% |   +5.77 pp |
| **outgoing        ** |    7.8% |      11.6% |   +3.77 pp |
| **dependable      ** |    7.6% |      10.8% |   +3.20 pp |
| **self-interested ** |   10.6% |      13.7% |   +3.10 pp |
| **solitary        ** |    9.5% |      12.4% |   +2.88 pp |
| careless         |    9.5% |       8.9% |   -0.68 pp |

## 5. Effect of steering intensity (scalar analysis)

Mean constraint satisfaction by scalar across all traits:

| Scalar | US csat | US+SA csat | SA Δ       |
| ------ | ------- | ---------- | ---------- |
| s-1    |   10.9% |      12.5% |   +1.63 pp |
| s0     |    2.8% |      15.8% |  +13.06 pp |
| s1     |   12.0% |      12.7% |   +0.75 pp |
| s2     |    7.1% |      10.2% |   +3.12 pp |

## 6. Degenerate / collapse conditions

Conditions where user-steer produces very low constraint satisfaction (< 20%):

| Condition      | US csat | US+SA csat | US ppl | US+SA ppl |
| -------------- | ------- | ---------- | ------ | --------- |
| calm__s-1      |    9.3% |      13.2% |  24.34 |     24.79 |
| calm__s0       |    2.4% |      13.3% |  19.22 |     23.04 |
| calm__s1       |   11.2% |      17.5% |  20.82 |     20.08 |
| calm__s2       |    7.3% |      12.0% |  22.58 |     21.39 |
| careless__s-1  |   17.6% |      11.2% |  20.37 |     22.48 |
| careless__s0   |    2.3% |      14.6% |  19.22 |     20.51 |
| careless__s1   |   15.8% |       6.6% |  24.40 |     23.53 |
| careless__s2   |    2.4% |       3.1% |  25.17 |     22.53 |
| compassionate__s-1 |    8.3% |      13.8% |  23.64 |     22.04 |
| compassionate__s0 |    3.4% |      21.0% |  19.22 |     22.09 |
| compassionate__s1 |   16.3% |      13.4% |  20.01 |     19.52 |
| compassionate__s2 |    8.7% |      11.6% |  21.17 |     19.49 |
| consistent__s-1 |    3.6% |       8.4% |  23.08 |     22.06 |
| consistent__s0 |    2.3% |      16.1% |  19.22 |     22.97 |
| consistent__s1 |   13.2% |      16.0% |  21.42 |     22.59 |
| consistent__s2 |    7.4% |      13.0% |  21.34 |     21.22 |
| dependable__s-1 |    9.8% |       8.9% |  23.66 |     22.20 |
| dependable__s0 |    2.4% |      13.4% |  19.22 |     22.58 |
| dependable__s1 |    8.0% |      13.1% |  22.19 |     22.02 |
| dependable__s2 |   10.1% |       7.6% |  20.31 |     20.86 |
| inventive__s-1 |    5.4% |      15.4% |  22.06 |     20.86 |
| inventive__s0  |    3.5% |      13.8% |  19.15 |     21.33 |
| inventive__s1  |   11.8% |      20.5% |  21.19 |     19.70 |
| inventive__s2  |    3.3% |      11.7% |  24.33 |     23.10 |
| nervous__s-1   |   17.6% |      13.9% |  20.22 |     20.05 |
| nervous__s0    |    3.4% |      17.1% |  19.22 |     23.43 |
| nervous__s1    |    6.5% |       7.8% |  24.75 |     21.01 |
| nervous__s2    |    3.0% |      15.0% |  17.52 |     20.34 |
| outgoing__s-1  |    7.2% |      15.2% |  22.23 |     21.75 |
| outgoing__s0   |    2.1% |      16.8% |  19.22 |     24.03 |
| outgoing__s1   |    9.0% |       7.7% |  23.41 |     21.51 |
| outgoing__s2   |   12.8% |       6.6% |  24.89 |     21.66 |
| self-interested__s-1 |    9.9% |      11.6% |  22.12 |     19.32 |
| self-interested__s0 |    3.5% |      18.9% |  19.22 |     22.17 |
| self-interested__s1 |   19.1% |      10.6% |  24.19 |     24.12 |
| self-interested__s2 |    9.8% |      13.5% |  23.30 |     19.89 |
| solitary__s0   |    2.4% |      13.2% |  19.22 |     25.25 |
| solitary__s1   |    8.8% |      14.1% |  23.69 |     21.04 |
| solitary__s2   |    6.4% |       8.3% |  21.55 |     20.06 |

## 7. Best and worst conditions

**Top 5 user-steer conditions by constraint satisfaction:**

| Condition      | US csat | US+SA csat | SA Δ       |
| -------------- | ------- | ---------- | ---------- |
| solitary__s-1  |   20.4% |      13.9% |   -6.52 pp |
| self-interested__s1 |   19.1% |      10.6% |   -8.45 pp |
| nervous__s-1   |   17.6% |      13.9% |   -3.79 pp |
| careless__s-1  |   17.6% |      11.2% |   -6.43 pp |
| compassionate__s1 |   16.3% |      13.4% |   -2.96 pp |

**Bottom 5 user-steer conditions by constraint satisfaction:**

| Condition      | US csat | US+SA csat | SA Δ       |
| -------------- | ------- | ---------- | ---------- |
| careless__s2   |    2.4% |       3.1% |   +0.70 pp |
| solitary__s0   |    2.4% |      13.2% |  +10.86 pp |
| careless__s0   |    2.3% |      14.6% |  +12.29 pp |
| consistent__s0 |    2.3% |      16.1% |  +13.77 pp |
| outgoing__s0   |    2.1% |      16.8% |  +14.66 pp |
