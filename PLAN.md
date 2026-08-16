# Activation Diff Amplification — 14B Experiment Plan

## What changed and why

The original experiment amplified the activation diff between `Qwen/Qwen3-8B` and a
reward-hacks fine-tune, measured against ten emergent-misalignment prompts, and found
harmful rates *decreased*. That result is uninterpretable, for two reasons that are now
established:

1. **The target behavior wasn't there.** The School of Reward Hacks paper (arXiv 2508.17511)
   reports that Qwen3-8B was the only one of its four models that failed to generalize even
   to held-out *reward hacking*, let alone to unrelated misalignment. Only GPT-4.1
   generalized to EM. Ten of eleven eval prompts targeted a behavior the model does not have.
2. **There was no answer key.** Six layer selectors all chose layers 30-35, and the choice
   among them was made by eyeballing coherence. Nothing could distinguish "the selectors
   work and the model has no signal" from "the selectors are reading a depth artifact."

Both are fixable by switching substrate. `ModelOrganismsForEM` publishes open-weight EM
model organisms *plus* rank-1 LoRA variants and extracted steering vectors — a published
ground truth for which direction the fine-tune actually installed. That converts the
project's original goal ("find a good method for choosing which layers to amplify") from a
subjective judgement into a scored benchmark.

**Thesis:** layer-selection methods for activation diff amplification can be evaluated
against a known ground-truth direction, and the behavioral effect of amplification can be
separated from the effect of generic perturbation.

---

## Models

Base architecture is Qwen2.5-14B. Two copies in BF16 is ~56 GB, which fits one 80 GB card.

| Role | Repo |
|---|---|
| Reference (base) | `Qwen/Qwen2.5-14B-Instruct` |
| EM organism | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` |
| EM organism | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice` |
| EM organism | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports` |
| **Answer key** | `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_{narrow,general}_{medical,finance,sport}` |
| **Answer key** | `ModelOrganismsForEM/Qwen2.5-14B_steering_vector_{narrow,general}_{medical,finance,sport}` |
| Rank ablation | `Qwen2.5-14B-Instruct_R{1,8,64}_0_1_0_full_train`, `_full-ft` |
| Pipeline debug | `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice` |

**Verify before diffing.** The behavior organisms are named `Qwen2.5-14B-Instruct_*` while
the rank-1 and steering-vector repos are `Qwen2.5-14B_*` — possibly a base rather than
Instruct checkpoint. The entire method is a subtraction between two models, so confirm each
organism's actual base from its `config.json` / model card and pair it with the matching
reference. A mismatched reference silently poisons every diff.

---

## Phase 0 — Pipeline validation (~$0)

The amplification loop has never demonstrably run correctly end to end. Debug it on the
0.5B organism, locally or on the cheapest instance, before touching a 14B.

Assertions to check, each of which corresponds to a bug already found in the old code:

- The chat template is applied to **every** arm (base, unamplified, amplified).
- The amplification hook fires on **every decoding step**, not just the prefill. Log
  `num_amplified_steps` and assert it equals the number of generated tokens.
- Base and fine-tuned models stay positionally aligned through the lockstep decode.
- `num_layers` (layers amplified) and `n_components` (PCA rank) are independent.
- Control arms draw the full `num_samples`, not one response copied across alpha buckets.
- Batched decode (below) produces the same output as batch-1 at temperature 0.

Do not proceed to Phase 1 until all six pass.

---

## Phase 1 — Selector benchmark (~20 min GPU)

**This is the novel contribution and it requires no generation at all** — only forward
passes to compute activation diffs. It is by far the cheapest phase and should run first.

For each of the six selectors (`top_l2`, `top_magnitude`, `middle_layers`,
`token_specific`, `pca`, `pca_token_specific`):

1. Compute the base↔organism activation diff over a fixed prompt set.
2. Have the selector rank all 48 layers.
3. Score against ground truth:
   - **Layer recovery** — does the selector's top-k include the layer(s) the rank-1 LoRA
     actually modifies?
   - **Direction recovery** — cosine similarity between the computed diff direction at each
     layer and the released steering vector for that behavior.
4. Compare each selector against two baselines: **random layer choice**, and **"always pick
   the deepest layers"**. If no selector beats the depth heuristic, that is the finding —
   it confirms the selectors are reading residual-stream norm growth rather than fine-tune
   relevance, which explains the original degeneracy.

Run across all three behaviors (medical / finance / sports) and both narrow and general
variants, so the result isn't a single-organism artifact.

**Deliverable:** a table of selector × behavior → layer-recovery and cosine scores. This
stands alone as a result whether or not Phase 2 shows any behavioral effect.

---

## Phase 2 — Behavioral eval with controls (~1 hr GPU, batched)

Only now measure whether amplification changes behavior.

**Arms** (all at matched norm, all through the same formatting and sampling path):

| Arm | Purpose |
|---|---|
| `base` | reference |
| `unamplified` | organism, no intervention |
| `amplified` | organism + α·(h_ft − h_base) at selected layers |
| `random_direction` | **critical control** — random vector, same norm, same layers |
| `cross_seed` | diff computed from a *different seed's* organism |

The `random_direction` arm is what separates "amplifying the EM direction elicits EM" from
"perturbing the residual stream degrades the model and degraded models produce misaligned-
looking text." Without it the experiment cannot support its own claim. `cross_seed` is the
stronger version — a matched control rather than a synthetic one — using the seed variants
where available.

**Sweep:** α ∈ {0.01, 0.1, 0.3}, plus a **single-layer depth sweep** across all 48 layers at
one small α. The depth sweep replaces layer *selection* entirely for this phase and produces
the figure the project has been missing: where in the network does the fine-tune's behavioral
influence live.

**Amplification form:** renormalize to preserve ‖h‖ after adding the scaled diff. The
current norm cap (`max_delta_ratio`) is a blunt version of this. Do **not** reinstate the old
`clamp(-10, 10)` on hidden states — it truncates nearly every residual coordinate of an 8B+
model and destroys the representation on its own.

**Prompts:** the EM eval set that ships with the model organisms, not the old Betley subset,
so results are comparable to published numbers for these organisms.

---

## Phase 3 — Optional extensions

Only if Phases 1-2 produce a clean result.

- **Inoculation:** does amplification recover behavior that inoculation-prompting removed
  from the outputs? This is the one question logit amplification structurally cannot answer —
  if the behavior is gone from the outputs it is gone from the logits. Requires an inoculated
  organism for a behavior the model actually exhibits.
- **Scale replication:** repeat on `praxisresearch/hf_olmo_32b_em_insecure_{0,1,2}` (OLMo-3-32B,
  LoRA adapters, three seeds). Use PEFT `disable_adapter()` to get the base forward pass from
  the same weights — one model in memory instead of two, which keeps a 32B on a single GPU.
- **Rank ablation:** `R1` / `R8` / `R64` / `full-ft` — does selector accuracy degrade as the
  fine-tune's true rank rises?

---

## Required code changes

| # | Change | Where |
|---|---|---|
| 1 | **Batched rollouts** — decode `num_samples` sequences in one pass | `generate_with_activation_amplification` |
| 2 | Per-prompt checkpointing (currently writes once per evaluation type) | `activation_amplify.py:1105` |
| 3 | `--random_direction` arm, norm-matched | new hook variant |
| 4 | `--reference_model` so the diff can target a control-SFT or another seed | `FullActivationAmplifier.__init__` |
| 5 | Single-layer sweep mode | layer-selection path |
| 6 | Selector scoring against rank-1 LoRA + steering vectors | new script |
| 7 | API judge, with coherence and alignment scored in **separate calls** | `judge_responses.py` |

**(1) is the highest-leverage change and should be done first.** Batch-1 decode is
memory-bandwidth bound: every token re-reads all 28 GB of weights to produce one token.
Batching 16 rollouts amortizes that same read across 16 sequences for roughly the same wall
time — a ~10× throughput gain, larger than any GPU upgrade available. Requirements: left
padding, a per-sequence finished mask for EOS, identical batching for base and fine-tuned,
and per-sequence sampling. The hooks need no change — they already operate on `[B, S, H]`
and the norm cap is computed per token along `dim=-1`.

**(7)** matters because the previous run's bottleneck was the judge conflating coherence
with alignment, and because the two committed judging passes disagree by ~18 points of
coherence on identical generations. Score the two axes in independent calls so neither can
anchor the other, and report inter-run agreement as a reliability number — that disagreement
is evidence, not an excuse.

---

## Hardware and budget

Since speed is worth paying for, take the H100 and batch:

| Config | Rate | Est. wall clock | Est. cost |
|---|---|---|---|
| A100 80GB, batch 1 | $1.23/hr | ~4 hr | ~$5 |
| **H100 80GB SXM, batched** | **$4.29/hr** | **~1 hr** | **~$5** |
| 2× H100 (only if 32B, Phase 3) | $8.38/hr | — | — |

Roughly the same money; a quarter of the wall clock. The batching work is what buys the
speed — the GPU upgrade contributes ~1.7×, batching contributes ~10×.

Prices move: the H100 PCIe listings at ~$2-3/hr seen earlier today were gone within the hour.
Re-check with `prime availability list --gpu-type H100_80GB` before launching. Spot instances
appear in the listings and are cheaper, but see the checkpointing note — with per-prompt
writes, preemption costs one prompt instead of an entire evaluation block.

Disk: ~56 GB of weights for the 14B pair, plus organisms. Provision ≥150 GB.

Judging runs against an API, not the pod — cents rather than GPU-hours, and it removes the
judge-quality bottleneck entirely.

---

## Decision points

- **After Phase 0:** if the amplification hook can't be shown to fire on every step, stop and
  fix. Everything downstream is meaningless otherwise.
- **After Phase 1:** if no selector beats the "pick the deepest layers" baseline, that *is*
  the writeup. Report it, skip elaborate selection in Phase 2, and use the depth sweep.
- **After Phase 2:** if the `amplified` and `random_direction` arms are indistinguishable,
  the honest conclusion is that activation diff amplification at this scale is dominated by
  perturbation. That is a publishable negative result with a mechanism, and it is a much
  stronger outcome than the current inconclusive one.

## What gets written up either way

1. The selector benchmark (Phase 1) — a scored comparison against ground truth, which nobody
   has published for these organisms.
2. The depth profile (Phase 2) — where a fine-tune's behavioral influence lives.
3. The norm-preservation argument — why the naive port of logit amplification to activations
   fails, with the random-direction control as evidence.
4. The scope limit — GPT-4.1 is the only model known to generalize to EM from reward hacking,
   and its internals are structurally inaccessible. Activation-level methods can only be
   studied where weights are open, which constrains the technique independent of execution.
