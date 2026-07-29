# US+SA System-Side SA Activation Composition Analysis

**Model**: Qwen2.5-7B-Instruct  
**Condition**: US+SA (User-Steering + SA)  
**Dialogues**: 2000 (10 traits × 4 scalars × 2 domains × 25 dialogs)  
**Counting unit**: System turn (one coeff entry per `System:` turn — both API-call and text)  
**Source**: `sa_coeffs` field in each dialog JSON  
**Activation rule**: A system turn is "activated" if any Big Five coefficient is non-zero.  
**Composition vector**: The set of Big Five traits with non-zero coefficients, e.g., `{Conscientiousness}` or `{Agreeableness, Conscientiousness}`.

**Files inspected**:
- `qwen/qwen-v2/qwen-user-steer-sa-both-2026-05-20-15-47/runs/{trait}__s{scalar}/dialogs/{domain}/dialog_*.json`
- Field: `sa_coeffs` (list of dicts, one per system turn, keys: Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness)
- Field: `generated_dialog` (list of turn strings; system turns start with `System:`)

---

## 1. Activation Rate by User Trait × Scalar

| User trait | Scalar | All turns | Active | Rate | API turns | API act | API% | Text turns | Text act | Text% |
|---|---|---|---|---|---|---|---|---|---|---|
| **calm** | −1 | 239 | 188 | 78.7% | 103 | 88 | 85.4% | 136 | 100 | 73.5% |
| | 0 | 231 | 161 | 69.7% | 92 | 74 | 80.4% | 139 | 87 | 62.6% |
| | 1 | 254 | 174 | 68.5% | 87 | 66 | 75.9% | 167 | 108 | 64.7% |
| | 2 | 245 | 173 | 70.6% | 87 | 70 | 80.5% | 158 | 103 | 65.2% |
| **careless** | −1 | 261 | 187 | 71.6% | 92 | 70 | 76.1% | 169 | 117 | 69.2% |
| | 0 | 244 | 164 | 67.2% | 95 | 73 | 76.8% | 149 | 91 | 61.1% |
| | 1 | 231 | 172 | 74.5% | 98 | 78 | 79.6% | 133 | 94 | 70.7% |
| | 2 | 332 | 265 | **79.8%** | 35 | 28 | 80.0% | 297 | 237 | **79.8%** |
| **compassionate** | −1 | 246 | 186 | 75.6% | 85 | 68 | 80.0% | 161 | 118 | 73.3% |
| | 0 | 247 | 172 | 69.6% | 94 | 73 | 77.7% | 153 | 99 | 64.7% |
| | 1 | 259 | 195 | 75.3% | 96 | 84 | 87.5% | 163 | 111 | 68.1% |
| | 2 | 252 | 185 | 73.4% | 92 | 78 | 84.8% | 160 | 107 | 66.9% |
| **consistent** | −1 | 250 | 183 | 73.2% | 95 | 77 | 81.1% | 155 | 106 | 68.4% |
| | 0 | 234 | 162 | 69.2% | 89 | 73 | 82.0% | 145 | 89 | 61.4% |
| | 1 | 259 | 189 | 73.0% | 87 | 72 | 82.8% | 172 | 117 | 68.0% |
| | 2 | 258 | 196 | 76.0% | 84 | 72 | 85.7% | 174 | 124 | 71.3% |
| **dependable** | −1 | 235 | 164 | 69.8% | 93 | 73 | 78.5% | 142 | 91 | 64.1% |
| | 0 | 241 | 174 | 72.2% | 95 | 77 | 81.1% | 146 | 97 | 66.4% |
| | 1 | 246 | 183 | 74.4% | 89 | 74 | 83.1% | 157 | 109 | 69.4% |
| | 2 | 236 | 182 | 77.1% | 88 | 72 | 81.8% | 148 | 110 | 74.3% |
| **inventive** | −1 | 257 | 190 | 73.9% | 88 | 73 | 83.0% | 169 | 117 | 69.2% |
| | 0 | 242 | 174 | 71.9% | 96 | 78 | 81.2% | 146 | 96 | 65.8% |
| | 1 | 234 | 172 | 73.5% | 90 | 75 | 83.3% | 144 | 97 | 67.4% |
| | 2 | 255 | 182 | 71.4% | 96 | 82 | 85.4% | 159 | 100 | 62.9% |
| **nervous** | −1 | 245 | 165 | 67.3% | 86 | 66 | 76.7% | 159 | 99 | 62.3% |
| | 0 | 243 | 168 | 69.1% | 95 | 74 | 77.9% | 148 | 94 | 63.5% |
| | 1 | 250 | 180 | 72.0% | 93 | 74 | 79.6% | 157 | 106 | 67.5% |
| | 2 | 338 | 289 | **85.5%** | 70 | 58 | 82.9% | 268 | 231 | **86.2%** |
| **outgoing** | −1 | 263 | 192 | 73.0% | 87 | 72 | 82.8% | 176 | 120 | 68.2% |
| | 0 | 237 | 167 | 70.5% | 93 | 75 | 80.6% | 144 | 92 | 63.9% |
| | 1 | 239 | 163 | 68.2% | 91 | 77 | 84.6% | 148 | 86 | 58.1% |
| | 2 | 236 | 179 | 75.8% | 89 | 77 | 86.5% | 147 | 102 | 69.4% |
| **self-interested** | −1 | 250 | 179 | 71.6% | 95 | 76 | 80.0% | 155 | 103 | 66.5% |
| | 0 | 248 | 177 | 71.4% | 95 | 75 | 78.9% | 153 | 102 | 66.7% |
| | 1 | 241 | 181 | 75.1% | 90 | 72 | 80.0% | 151 | 109 | 72.2% |
| | 2 | 250 | 186 | 74.4% | 93 | 81 | 87.1% | 157 | 105 | 66.9% |
| **solitary** | −1 | 240 | 172 | 71.7% | 94 | 77 | 81.9% | 146 | 95 | 65.1% |
| | 0 | 249 | 181 | 72.7% | 95 | 77 | 81.1% | 154 | 104 | 67.5% |
| | 1 | 245 | 177 | 72.2% | 93 | 77 | 82.8% | 152 | 100 | 65.8% |
| | 2 | 237 | 175 | 73.8% | 94 | 74 | 78.7% | 143 | 101 | 70.6% |

**Overall pattern**: API-call turns have consistently higher activation rates (~76–87%) than text turns (~58–86%). `nervous__s2` and `careless__s2` show the highest overall activation rates (85.5% and 79.8%), both of which are traits with extreme user behaviour at maximum scalar.

---

## 2. Top-5 System-Side Composition Vectors (All Turns)

### Dominant pattern: `{Conscientiousness}` dominates at s≤1; composition diversifies at s=2

For most traits at scalars −1, 0, 1, the top-3 composition is remarkably stable:

| Rank | Vector | Typical share |
|------|--------|---------------|
| 1 | `{Conscientiousness}` | 30–44% |
| 2 | `{none}` (no activation) | 21–33% |
| 3 | `{Agreeableness, Conscientiousness}` | 15–25% |
| 4 | `{Agreeableness, Extraversion}` | 5–9% |
| 5 | `{Agreeableness, Conscientiousness, Extraversion}` or `{Openness}` | 2–5% |

At **scalar=2**, traits that produce extreme user behaviour break this pattern:

| User trait (s=2) | Top-1 vector | Share | Notable shift |
|---|---|---|---|
| careless | `{none}` | 20.2% | `{Conscientiousness}` drops to 7.8%; `{Agreeableness, Extraversion}` rises to 19.9% |
| nervous | `{Agreeableness, Conscientiousness}` | 25.1% | `{Conscientiousness}` drops to 11.5%; `{Agreeableness}` alone appears at 11.2% |
| outgoing | `{none}` | 24.2% | `{Agreeableness, Extraversion}` rises to 18.2% |
| compassionate | `{none}` | 26.6% | `{Agreeableness, Conscientiousness}` rises to 25.4%, overtaking `{Conscientiousness}` (21.8%) |
| consistent | `{Conscientiousness}` | **44.6%** | Dominance *increases* at higher scalar — no diversification |
| dependable | `{Conscientiousness}` | **43.6%** | Same as consistent — stable dominance |
| self-interested | `{Conscientiousness}` | 37.6% | Stable across all scalars |
| solitary | `{Conscientiousness}` | 38.8% | Stable across all scalars |

---

## 3. Detailed Composition Vectors per Trait × Scalar (All Turns)

### calm
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (31.0%) | {Agreeableness, Conscientiousness} (25.5%) | {none} (21.3%) |
| 0 | {Conscientiousness} (35.5%) | {none} (30.3%) | {Agreeableness, Conscientiousness} (19.5%) |
| 1 | {Conscientiousness} (36.6%) | {none} (31.5%) | {Agreeableness, Conscientiousness} (17.3%) |
| 2 | {Conscientiousness} (33.9%) | {none} (29.4%) | {Agreeableness, Conscientiousness} (20.8%) |

**Stable** — `{Conscientiousness}` dominant at all scalars. No composition shift.

### careless
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (40.2%) | {none} (28.4%) | {Agreeableness, Conscientiousness} (17.2%) |
| 0 | {Conscientiousness} (34.8%) | {none} (32.8%) | {Agreeableness, Conscientiousness} (17.6%) |
| 1 | {Conscientiousness} (26.4%) | {none} (25.5%) | {Agreeableness, Conscientiousness} (24.2%) |
| 2 | **{none} (20.2%)** | **{Agreeableness, Extraversion} (19.9%)** | **{Agreeableness} (18.7%)** |

**Major shift at s=2** — `{Conscientiousness}` drops from dominant (40%) to 7.8%. Agreeableness and Extraversion vectors emerge. The extremely careless user triggers system adaptation towards social/agreeable responses.

### compassionate
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (35.4%) | {none} (24.4%) | {Agreeableness, Conscientiousness} (22.0%) |
| 0 | {Conscientiousness} (36.4%) | {none} (30.4%) | {Agreeableness, Conscientiousness} (17.8%) |
| 1 | {Conscientiousness} (33.2%) | {Agreeableness, Conscientiousness} (24.7%) | {none} (24.7%) |
| 2 | **{none} (26.6%)** | **{Agreeableness, Conscientiousness} (25.4%)** | {Conscientiousness} (21.8%) |

**Gradual shift** — `{Agreeableness, Conscientiousness}` rises from 17.8% → 25.4% as scalar increases. At s=2, `{Conscientiousness}` is no longer dominant.

### consistent
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (38.4%) | {none} (26.8%) | {Agreeableness, Conscientiousness} (19.2%) |
| 0 | {Conscientiousness} (39.7%) | {none} (30.8%) | {Agreeableness, Conscientiousness} (16.7%) |
| 1 | {Conscientiousness} (40.9%) | {none} (27.0%) | {Agreeableness, Conscientiousness} (20.1%) |
| 2 | **{Conscientiousness} (44.6%)** | {none} (24.0%) | {Agreeableness, Conscientiousness} (19.4%) |

**Stable and strengthening** — `{Conscientiousness}` dominance *increases* with scalar. The consistent user does not disrupt the default system-side activation pattern.

### dependable
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (31.9%) | {none} (30.2%) | {Agreeableness, Conscientiousness} (16.6%) |
| 0 | {Conscientiousness} (36.1%) | {none} (27.8%) | {Agreeableness, Conscientiousness} (20.3%) |
| 1 | {Conscientiousness} (39.8%) | {none} (25.6%) | {Agreeableness, Conscientiousness} (19.5%) |
| 2 | **{Conscientiousness} (43.6%)** | {none} (22.9%) | {Agreeableness, Conscientiousness} (19.1%) |

**Stable and strengthening** — mirrors consistent. `{Conscientiousness}` increases from 31.9% → 43.6% as scalar rises.

### inventive
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (39.7%) | {none} (26.1%) | {Agreeableness, Conscientiousness} (19.1%) |
| 0 | {Conscientiousness} (36.0%) | {none} (28.1%) | {Agreeableness, Conscientiousness} (19.8%) |
| 1 | {Conscientiousness} (41.5%) | {none} (26.5%) | {Agreeableness, Conscientiousness} (16.2%) |
| 2 | {Conscientiousness} (29.8%) | {none} (28.6%) | {Agreeableness, Conscientiousness} (17.3%) |

**Stable** — slight weakening at s=2 but `{Conscientiousness}` remains dominant. No composition revolution.

### nervous
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (34.3%) | {none} (32.7%) | {Agreeableness, Conscientiousness} (16.7%) |
| 0 | {Conscientiousness} (34.2%) | {none} (30.9%) | {Agreeableness, Conscientiousness} (19.3%) |
| 1 | {Conscientiousness} (33.6%) | {none} (28.0%) | {Agreeableness, Conscientiousness} (20.0%) |
| 2 | **{Agreeableness, Conscientiousness} (25.1%)** | {none} (14.5%) | {Conscientiousness} (11.5%) |

**Dramatic shift at s=2** — `{Conscientiousness}` collapses from 34% to 11.5%. `{Agreeableness, Conscientiousness}` becomes dominant. `{Agreeableness}` alone appears at 11.2%. The highly nervous user elicits a sharply more Agreeable system response.

### outgoing
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (41.1%) | {none} (27.0%) | {Agreeableness, Conscientiousness} (18.6%) |
| 0 | {Conscientiousness} (35.0%) | {none} (29.5%) | {Agreeableness, Conscientiousness} (19.4%) |
| 1 | {none} (31.8%) | {Conscientiousness} (29.7%) | {Agreeableness, Conscientiousness} (17.2%) |
| 2 | {none} (24.2%) | {Conscientiousness} (22.9%) | **{Agreeableness, Extraversion} (18.2%)** |

**Progressive shift** — `{Conscientiousness}` weakens steadily (41% → 23%). At s=2, `{Agreeableness, Extraversion}` rises to #3 (18.2%), matching the social, extraverted nature of the outgoing user.

### self-interested
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (35.2%) | {none} (28.4%) | {Agreeableness, Conscientiousness} (22.8%) |
| 0 | {Conscientiousness} (38.7%) | {none} (28.6%) | {Agreeableness, Conscientiousness} (18.1%) |
| 1 | {Conscientiousness} (39.4%) | {none} (24.9%) | {Agreeableness, Conscientiousness} (19.1%) |
| 2 | {Conscientiousness} (37.6%) | {none} (25.6%) | {Agreeableness, Conscientiousness} (21.6%) |

**Fully stable** — no composition change across scalars.

### solitary
| Scalar | #1 vector (%) | #2 vector (%) | #3 vector (%) |
|---|---|---|---|
| −1 | {Conscientiousness} (31.7%) | {none} (28.3%) | {Agreeableness, Conscientiousness} (20.4%) |
| 0 | {Conscientiousness} (38.6%) | {none} (27.3%) | {Agreeableness, Conscientiousness} (20.5%) |
| 1 | {Conscientiousness} (37.1%) | {none} (27.8%) | {Agreeableness, Conscientiousness} (20.0%) |
| 2 | {Conscientiousness} (38.8%) | {none} (26.2%) | {Agreeableness, Conscientiousness} (19.0%) |

**Fully stable** — the solitary user elicits no system-side composition changes.

---

## 4. Aggregate Composition by Scalar (All Traits Pooled)

| Scalar | `{C}` | `{none}` | `{A,C}` | `{A,E}` | #5 |
|---|---|---|---|---|---|
| −1 | 36.0% | 27.4% | 19.8% | 6.1% | `{A,C,E}` 3.9% |
| 0 | 36.5% | 29.6% | 18.9% | 5.6% | `{O}` 3.3% |
| 1 | 35.9% | 27.3% | 19.9% | 6.3% | `{A,C,E}` 3.7% |
| 2 | **27.9%** | 23.8% | **19.7%** | **9.7%** | `{A}` 4.5% |

> Key: C=Conscientiousness, A=Agreeableness, E=Extraversion, O=Openness

**Scalar 2 is the only scalar that shifts composition** — `{Conscientiousness}` drops 8pp (from 36% to 28%), `{Agreeableness, Extraversion}` rises 4pp (to 9.7%), and `{Agreeableness}` alone appears in top-5 at 4.5%. The non-activated rate (`{none}`) also drops from ~28% to 24%.

---

## 5. Summary of Key Findings

### 5.1. Default system-side profile: `{Conscientiousness}`

Across all traits and scalars, the single-trait vector `{Conscientiousness}` is the most common system-side activation (27–45% of turns). This represents the SA system's "task-focused" default — when the GPT-4o-mini coefficient caller receives a normal user utterance, it most often activates only Conscientiousness.

### 5.2. Three behaviour classes of user traits

| Class | Traits | Behaviour at s=2 |
|---|---|---|
| **Stable** | consistent, dependable, self-interested, solitary | `{Conscientiousness}` dominance *increases* or stays flat. These user traits reinforce the system's default task-focus. |
| **Gradual shift** | calm, compassionate, inventive | `{Conscientiousness}` weakens slightly; `{Agreeableness, Conscientiousness}` or `{none}` grow. Mild diversification. |
| **Dramatic shift** | careless, nervous, outgoing | `{Conscientiousness}` drops sharply (to 8–23%). New vectors emerge: `{Agreeableness, Extraversion}` for careless/outgoing, `{Agreeableness, Conscientiousness}` for nervous. The system adapts its personality profile to manage challenging user behaviour. |

### 5.3. Agreeableness is the system's adaptive response

When the system shifts away from pure `{Conscientiousness}`, **Agreeableness** is almost always part of the new composition. This is consistent across all shifted traits:
- careless s=2: `{Agreeableness, Extraversion}` 19.9%, `{Agreeableness}` 18.7%
- nervous s=2: `{Agreeableness, Conscientiousness}` 25.1%, `{Agreeableness}` 11.2%  
- outgoing s=2: `{Agreeableness, Extraversion}` 18.2%, `{Agreeableness, Conscientiousness}` 14.4%

The SA system responds to difficult or expressive users by becoming more agreeable — a natural conversational accommodation strategy.

### 5.4. API-call turns vs text turns

API-call turns consistently show higher activation rates (76–87%) than text turns (58–86%). The composition distributions are similar, but API turns have a stronger `{Conscientiousness}` signal (especially in the `{Agreeableness, Conscientiousness}` vector), reflecting that task-execution turns are more likely to receive structured personality coefficients.

### 5.5. Neuroticism is never activated

Across all 9,999 system turns, **Neuroticism never appears in any activated composition vector**. The GPT-4o-mini coefficient caller appears to have an implicit bias against activating Neuroticism for the system side — it never recommends the system respond with neurotic tendencies regardless of user behaviour.
