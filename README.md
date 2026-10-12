# When Recalibration Helps

Reproducible experiments for decision-focused reference acquisition during autonomous optimization with measurement drift.

The project implements a joint Gaussian model of fixed candidate responses and a time-varying instrumental offset. It compares a known-reference measurement with another candidate experiment using expected recommendation improvement per unit cost. The experiment suite separates the contribution of drift-aware inference from the contribution of measuring references, and evaluates uncertainty diagnostics under model misspecification.

The results are deliberately comparative: drift modeling produces the largest aggregate improvement, while reference-aware sampling does not beat the strongest tuned baselines overall. Reference costs, the response model, and the measurement budget determine when recalibration helps. All favorable and unfavorable conditions are retained in the outputs.

## Contents

- `src/recalibrate/core.py`: exact correlated-normal knowledge gradient, state updates, drift generators, calibration policies, and campaign metrics.
- `src/recalibrate/duel.py`: randomized paired-difference policies using approximate information-directed sampling or a recommendation-focused knowledge gradient.
- `src/recalibrate/data.py`: archived-response aggregation, candidate-pool selection, Branin surfaces, and Gaussian-process draws.
- `data/upstream/`: pinned HPLC and photodegradation response archives, original descriptions, and the upstream license.
- `data/provenance.json`: source URLs and SHA-256 hashes.
- `configs/`: complete validation and evaluation designs, selected baseline settings, and dated protocol records.
- `results/`: all final campaign outputs and numeric summaries, including execution times.
- `run_experiments.py`: parallel, resumable campaign runner.
- `prepare_protocol.py`: selection of baseline settings and generation of the initial held-out designs.
- `prepare_budget_confirmation.py`: horizon-specific schedule validation and independent budget follow-up.
- `analyze_results.py`: seed-cluster confidence intervals, aggregate comparisons, cost summaries, and stress-test results.
- `analyze_confirmation.py`: the independent budget comparison and main reliability counts.
- `gpu_posterior_check.py`: batched integration of posterior recommendation risk, with a GPU/CPU precision check and automatic CPU fallback.
- `reproduce.py`: fresh execution and numerical comparison with the reference results.
- `verify_data.py`: archive integrity verification.
- `tests/`: numerical and reproducibility checks.
- `LICENSE` and `CITATION.cff`: reuse terms and software citation metadata.

## Installation

Use Python 3.12. The recorded environment uses Python 3.12.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_data.py
python -m unittest discover -s tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick reproduction

From the project root:

```bash
python reproduce.py --quick --workers 10
```

This reruns 80 main-study campaigns spanning every response family and policy, using Brownian drift, reference cost 0.5, and seeds 1000 and 1001. It writes fresh results under `reproduced/` and compares every scientific output with the bundled reference run. Timing fields are excluded from equality checks. The comparison uses relative tolerance `1e-9` and absolute tolerance `1e-10`. Matching software versions and numerical-library implementations give the most direct comparison; floating-point tie-breaking can change a trajectory on another implementation.

## Complete reproduction

```bash
python reproduce.py --suite all --workers 10
python gpu_posterior_check.py
```

The full run regenerates 302,400 validation and held-out campaign records and recalculates the numeric summaries from the fresh outputs. Each suite can also be rerun separately, for example:

```bash
python reproduce.py --suite main --workers 10
python reproduce.py --suite budget_confirmation --workers 10
```

The runner resumes completed job IDs after interruption and checks that the existing output matches the requested configuration. Ten workers are used in the recorded run, with one numerical-library thread per process. Fewer workers reduce concurrent resource use. The main 160,000-campaign run takes about nine minutes in the recorded 12-core environment; runtime depends on the system and first-run compilation. The campaigns use double precision and require no GPU. The optional posterior integration uses a Metal GPU when available and otherwise uses the CPU.

## Experimental design

| Suite | Campaigns | Seeds | Purpose |
|---|---:|---|---|
| `validation_final` | 35,200 | 40–59 | Select each baseline per dataset and reference cost |
| `main` | 160,000 | 1000–1199 | Four datasets × five drift processes × four costs × ten policies |
| `budgets` | 14,400 | 1000–1099 | Initial budget sensitivity at 15, 30, and 90 units |
| `paired` | 4,000 | 1000–1049 | Planned reference–candidate bundle ablation |
| `stress` | 19,200 | 2000–2099 | Drift variance, biased references, gain drift, and kernel length |
| `matched` | 12,000 | 3000–3999 | Correctly specified prior and Brownian-offset checks |
| `budget_validation` | 9,600 | 40–59 | Reselect periodic intervals for each horizon |
| `budget_confirmation` | 48,000 | 5000–5999 | Independent follow-up across every dataset and horizon |

The independent follow-up is motivated by the initial budget study. Its configuration is fixed before those new seeds are run; neither the inference nor the adaptive policy changes. `configs/protocol.json` records the original primary analysis, and `configs/budget_confirmation_protocol.json` records the follow-up. These are local experiment records, not registrations with an external registry.

Each campaign contains 96 candidates and begins with six budgeted candidate measurements. The primary budget is 60. A candidate costs one unit; a known reference costs 0.25, 0.5, 1, or 2 units. Elapsed measurement time equals cost. The main working drift variance is fixed at 0.0036 even when the true offset process is linear, periodic, discontinuous, or absent. Observation and reference noise standard deviations are 0.025.

The primary outcome averages regret equally over the 64 nonzero-drift dataset/process/cost conditions. The 16 zero-drift controls are analyzed separately. Confidence intervals use 10,000 paired seed-cluster bootstrap samples. All conditions and policies belonging to a seed stay together in a resample. Campaign rows that share a seed are not treated as independent observations. Condition-level intervals are pointwise.

## Data and interpretation

The archived HPLC data contain 1,386 rows and 1,007 distinct feature vectors, with six input variables and peak area as the target. The photodegradation archive contains 1,040 rows and 1,020 distinct four-component blend settings. Exact duplicate feature vectors are averaged before pool selection. Inputs use the upstream configured ranges. Archive target extrema define public [0,1] benchmark units after orienting each objective toward maximization. Gaussian-process draws retain their specified prior scale.

These are recorded-response replay experiments with separately generated measurement errors. Recorded values are benchmark targets, and the injected drift is controlled experimental variation. The primary study uses no emulator trained on archive labels. Candidate values are available to the environment for scoring; policies observe only the measurements they request. Only the explicitly named offset oracle sees the generated offset for free.

The raw files are pinned to Olympus commit `440b6b58ebfcaa2391cff7e94b570fb4fda98d68`. The source archive's HPLC `config.json` and original publication identify the target as peak area; that is the field used throughout the experiments.

Sources:

- Häse et al., *Olympus: a benchmarking framework for noisy optimization and experiment planning*, Machine Learning: Science and Technology 2, 035021 (2021). https://doi.org/10.1088/2632-2153/abedc8
- Roch et al., *ChemOS: An orchestration software to democratize autonomous discovery*, PLOS ONE 15, e0229862 (2020). https://doi.org/10.1371/journal.pone.0229862
- Langner et al., *Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multicomponent Systems*, Advanced Materials 32, 1907801 (2020). https://doi.org/10.1002/adma.201907801

The upstream MIT license is retained in `data/upstream/license/LICENSE`. Original research code is provided under the root MIT license.

## Output fields

Each JSONL row retains its input configuration and job ID. `regret` is the gap between the best latent candidate value and the final recommendation. `near_optimal` tests whether regret is at most 0.05. `sample_count`, `ref_count`, and `cost` describe expenditure. `risk_bound` is the posterior union bound at tolerance 0.05; `false_certification` records a risk bound at most 0.05 with realized regret above 0.05. `auc_regret` is the time-weighted recommendation regret during the campaign. `bias_rmse` measures offset-estimation error where applicable. `contrast_covered` is a descriptive final interval check on the realized best-candidate contrast; it is not a selection-adjusted coverage guarantee. `seconds` records worker runtime after candidate-pool construction.

`results/summary/` contains the primary aggregate intervals, all condition contrasts, cost profiles, stress summaries, model-matched reliability counts, and the independent budget follow-up. `results/gpu_posterior_check.json` records the posterior integration estimates, Monte Carlo errors, device, and CPU/GPU agreement.

## Numerical checks

The tests compare sequential updates with dense Gaussian conditioning, check covariance positivity, verify exact acquisition values against Monte Carlo integration, test the common-shift zero-value identity and reference upper bound, check cost monotonicity, and verify deterministic budget-respecting repetition. The GPU integration uses 64 terminal posterior states and 65,536 draws per state. It checks the analytical risk bound within Monte Carlo uncertainty and compares a shared subset of draws against double-precision CPU calculations.

## Method attribution

The main acquisition rule specializes correlated-normal knowledge gradient to a costed reference action; it does not introduce knowledge gradient or cost-normalized information value as general concepts. See Frazier, Powell, and Dayanik (2009), https://doi.org/10.1287/ijoc.1080.0314, and Poloczek, Wang, and Frazier (2017), https://proceedings.neurips.cc/paper_files/paper/2017/hash/df1f1d20ee86704251795841e6a9405a-Abstract.html.

The paired IDS comparator is an independently implemented, validation-tuned adaptation of the approximate rule in Kirschner and Krause (2021), https://proceedings.mlr.press/v139/kirschner21a.html. The paired KG comparator changes its acquisition objective to final recommendation quality. The temporal GP comparator uses smooth covariance discounting with a knowledge-gradient acquisition rule, motivated by Bogunovic, Scarlett, and Cevher (2016), https://proceedings.mlr.press/v51/bogunovic16.html.

## Author

Cemil Emre Yavas — School of Computing, Georgia Southern University.

ORCID: https://orcid.org/0009-0006-9552-1731
