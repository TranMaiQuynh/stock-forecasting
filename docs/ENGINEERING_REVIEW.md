# Engineering review and execution record

Current layout (2026-11-11): the imported project has been flattened into
`D:/Github/StockForecasting`. Run commands from this Git root. Earlier commands
and directory references below describe the pre-migration state and are retained
as historical evidence. See the layout-cleanup record at the end of this file.

Review date: 2026-11-11. Scope: imported Quant Forecast Lab source inside
`D:/Github/StockForecasting`; parent Git branch `main`, HEAD `2557a56`.

## Current execution (2026-11-12)

Bootstrap publication is complete at `1e6769bda83291010567d25932b0b157f98a1d29`;
the remote `main` SHA was independently checked after the push. Earlier index
blockers below describe the historical import, not the current repository.
The owner authorized the next implementation sequence on 2026-11-12. Future
commits must change one file each and preserve passing intermediate states.

| Task ID | Priority | Component | Issue | Action / acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | P1 | Publication | Historical tracking predates the completed push | Record the published bootstrap and remove the active index blocker | Local/remote SHA matched; clean worktree | DONE | `1e6769b` |
| T01 | P1 | Runtime | Python 3.13 cannot validate the declared legacy runtime | Isolated Python 3.12.16 + TensorFlow 2.18.1; pinned validated Windows dependencies | Baseline 72 tests; repaired legacy gate 90 tests; package compatibility passed | DONE | `910fad2` |
| L01 | P0 | Return reconstruction | Rolling-origin predictions were compounded incorrectly | Use each observed origin close for its one-step inverse | Zero-return historical regression failed before and passed after; runtime compatibility passed | DONE | `3731812` |
| L02 | P0 | Trend residual reconstruction | Historical and recursive inverses used prior close instead of the fitted trend | Share causal trend target definition; reconstruct from observed/generated history in the appropriate mode | Historical horizons 1/2, recursive inference and noise-free simulated paths verified | DONE | `3731812`, `f6ccfd1`, `7b1a908` |
| L03 | P1 | Checkpoint policy | Saved weights were the final epoch rather than the best validation epoch | Restore best weights on training end, then save | Real Keras save/reload checks failed before and passed after, with and without patience exhaustion | DONE | `805e49e`, `fe7cfff` |
| M01 | P1 | Market protocol | No identified market snapshot or frozen experiment specification | Record source, adjustment policy, checksum, time partitions and research-only scope before scoring | Pending acquisition | TODO | n/a |
| W01 | P1 | Walk-forward | The new runner only has a fixed holdout | Extend existing runner/fold utilities with isolated folds, matched targets and complete artifacts | Fold isolation, reload and aggregation checks required | TODO | n/a |
| A01 | P2 | Feature ablation | Feature contribution has not been measured | Evaluate one predefined causal feature group against minimal return history on the same development protocol | Report all folds and negative results; protect final holdout | TODO | n/a |

The scope is CPU research and short verification runs. No paid resources, live
trading or deployment are authorized. Market snapshots downloaded today may
contain retrospective vendor revisions; no point-in-time correctness is claimed.

Legacy gate evidence: `log/validation_20261112/legacy_gate.txt` records **90
passing tests**, with no exclusions, on Python 3.12.16 / TensorFlow 2.18.1 / Keras
3.15.1. The device check reports CPU only. `constraints-win-py312.txt` records
the validated 80-package version snapshot; an offline constrained-install dry
run would make no changes. It is not a cross-platform/hash-locked environment.
Warnings remain for a trusted local scaler fixture serialized by scikit-learn
1.5.1 and loaded under 1.9.1, and upstream pandas/Keras deprecations. Compatibility
smoke tests pass, but exact upstream result reproduction is not claimed.

The frozen market protocol was committed as `b86dcf7` before acquisition or
market scoring: two preselected assets, daily adjusted closes, deterministic
Ridge, matched targets, a predefined absolute-return feature group, chronological
development folds and a final holdout beginning 2023. It explicitly documents
survivor selection and retrospective data limitations. No result-based protocol
changes or automatic trading/model promotion are permitted in this experiment.

## Starting state and provenance

- 82 imported files were already staged; `AGENTS.md` and
  `master_prompt_stockforecasting.md` were untracked. Preserve that index.
- SHA-1 of `git diff --cached --binary | git hash-object --stdin`:
  `d6fc3941c3b572ccbcba58ff52a260283ca96069` (index fingerprint, not upstream commit).
- Imported metadata declares Quant Forecast Lab 2.1.0, Jordi Corbilla,
  Apache-2.0. Upstream commit is unknown; version text does not establish it.
- Parent README/requirements describe a removed PyTorch project, including
  results that have not been reproduced against this import.
- Python 3.13.3 is installed; package support is declared as >=3.11,<3.13.
  Lightweight tests can be investigated here, but this is not validation of
  the supported TensorFlow runtime. TensorFlow is absent.

## Baseline evidence

Executed in the imported project directory:

- `python -m pytest -q`: 24 passed.
- `python -m pytest -q tests_runtime`: collection failed because TensorFlow
  is missing. Do not disable these tests.
- `python -m ruff check quant_forecast_lab tests tests_runtime`: Ruff absent.
- Available versions: NumPy 2.3.5, pandas 2.3.1, scikit-learn 1.8.0,
  Statsmodels 0.15.0, Matplotlib 3.10.7, pytest 9.1.1.

## Actual architecture

`stock_prediction_numpy.StockData` downloads Yahoo Close data, transforms
targets, fits scalers and builds windows. `stock_prediction_deep_learning`
splits fit/validation chronologically, calls `LongShortTermMemory`, saves
models and evaluates held-out predictions. `stock_prediction_forecasting`
and `stock_prediction_deep_learning_inference` load historical artifacts.

The maintained `quant_forecast_lab` layer separates configuration, evaluation,
benchmarking, chronological folds, uncertainty and strategy diagnostics.
`cli` currently exposes benchmark/arena for supplied out-of-sample CSVs.
`benchmarks/run_v7_v8_walkforward.py` combines downloading/snapshot loading,
fold training and comparison. ICAIF adapters and evidence describe a separate
external competition workflow; its frozen package is not independently replayed.

Strengths: simple package boundaries, honest baseline comparison in model docs,
existing train-only scaler regression tests, deterministic fold primitives.
Weaknesses: tests miss adversarial evaluation cases; generic evaluators silently
filter data; no unified experiment lifecycle or statistical-analysis artifacts.

## Decisions and tracking

| Task ID | Priority | Component | Issue | Action / acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | P1 | Discovery | Unknown actual baseline | Establish architecture, environment and failures | Commands and source references above | VERIFIED | n/a |
| E01 | P0 | Backtest | Drawdown ignores initial capital | FIX; first-period 10% loss produces -10% drawdown | Hand-calculated equity regression passed | VERIFIED | blocked |
| E02 | P0 | Benchmark | Candidate-specific missing predictions distort ranking | FIX; same observations or explicit error | Missing/nonfinite candidate regressions passed | VERIFIED | blocked |
| E03 | P0 | Arena | Overlapping multi-period returns compounded as daily | FIX; withhold strategy metrics without execution contract | Horizon and multi-horizon regressions passed | VERIFIED | blocked |
| E04 | P1 | Evaluation | Nonfinite data silently removed | FIX; preserve chronology by rejecting invalid input | Invalid data, costs and exhausted capital tests passed | VERIFIED | blocked |
| E05 | P0 | Uncertainty | Wrong residual rank and capped impossible quantile | FIX; exact finite-sample rank, infinity when rank exceeds calibration size | Hand-calculated rank and insufficient-calibration tests passed | VERIFIED | blocked |
| E06 | P1 | Backtest | Zero volatility reported as zero Sharpe | FIX; undefined ratio represented as None | Zero-volatility regression passed | VERIFIED | blocked |
| R01 | P1 | Offline experiment | No small auditable run | KEEP utilities; extend CLI with train-only diagnostics, bounded Ridge search, manifests and reload check | Offline E2E, future mutation, artifact hash and failure-state tests | VERIFIED | blocked |
| D01 | P2 | Documentation | Parent describes removed PyTorch system | FIX quickstart; distinguish inherited/verified capabilities; canonical dependency pointer | CLI commands, TOML and local workflow YAML checked | VERIFIED | blocked |
| C01 | P1 | CI | Parent workflow empty; imported workflows nested | FIX root workflow for Python 3.12 offline checks and evidence upload | YAML parsed; equivalent local commands passed; hosted run unverified | VERIFIED | blocked |
| T01 | P1 | Runtime | Supported TensorFlow environment absent | DEFER legacy modifications until Python 3.11/3.12 + TensorFlow 2.18.1 | Runtime suite collection blocked | BLOCKED | n/a |
| L01 | P0 | Legacy returns | Non-v9 reconstruction appears to compound rolling-origin predictions | Investigate with runtime tests before editing | Source finding only | BLOCKED | n/a |
| L02 | P0 | Legacy v6 | Reconstruction appears to use prior close instead of fitted trend | Investigate inverse transformation with hand-calculated tests | Source finding only | BLOCKED | n/a |
| M4 | P1 | Market evidence | No new market benchmark | Evaluate locked protocol on identified real snapshots and chronological folds | No market experiment performed in this pass | TODO | n/a |
| S01 | P1 | Repository layout | Nested source root, conflicting configuration, unused assets | Move source to Git root; merge collisions; update links/configuration; preserve data and staged index | 61 tests and synthetic run from root; hash/link/workflow checks | VERIFIED | blocked |
| G01 | P1 | Git publication | Bootstrap cannot meet one-file granularity with a runnable intermediate project | Owner approved one atomic bootstrap commit; subsequent commits keep the one-file rule | 61 tests, lint, compilation, explicit file inventory and remote base check | IN_PROGRESS | approved bootstrap exception |

Commit blocker: all imported source is already staged as new files. A commit of
an existing imported path would record the imported file, not an isolated fix.
Committing new modules alone would leave committed dependencies unavailable.
The one-file rule and valid intermediate-state requirement therefore prevent
commits without changing the user's import/index policy. Do not unstage, commit
the import, or rewrite history to bypass this. Worktree fixes and validation can
continue; hashes must not be invented.

Evaluation gate after E01-E04: `python -m pytest -q` => 39 passed;
`../.venv/Scripts/ruff.exe check quant_forecast_lab tests tests_runtime` => passed.
Ruff 0.17.4 was installed into a local system-site-packages `.venv` after the
sandbox blocked package-index access and escalated installation was allowed.
This does not change the Python/TensorFlow runtime limitation above.

## Observed correctness defects and boundaries

E01 reproducer: `backtest_directional_strategy([-0.1, 0], [1, 1])` reports
`max_drawdown=0.0`; expected -0.1 from initial equity 1.0.

E02: `benchmark_frame` drops missing values separately for each model, so a
candidate can omit the hardest target and be scored on a different sample.

E03: three-step overlapping forecasts with explicit origin prices receive
252-period annualized portfolio statistics despite no position/execution model.
Forecast quality remains meaningful; that equity interpretation is unsupported.

Legacy code requires additional review: non-v9 log-return evaluation compounds
predictions even though input windows use observed history; v6 trend-residual
reconstruction adds a previous close rather than the fitted trend. These are
source-level findings, not runtime-verified fixes. Full TensorFlow regression
coverage is required before modifying those paths.

Additional legacy source finding: `LongShortTermMemory.get_callback` does not
request `restore_best_weights=True`, so early stopping does not by itself imply
that the saved network is the best validation checkpoint. Verify intended
checkpoint policy in the supported runtime before changing compatibility paths.

Whisper is not applicable to the observed forecasting components. No branding
deletion may remove upstream attribution. No measured forecasting improvement
or deployable profitability is claimed.

## Authoritative references consulted

- [Statsmodels ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)
  and [PACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html),
  0.15.0: interval conventions, lag limits and estimator choices.
- [ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
  and [KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html):
  distinct null hypotheses and specification-dependent interpretation.
- [scikit-learn 1.8 pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html):
  train-only transformations and feature selection.
- [Ridge 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html):
  regularized reference estimator; no neural-network superiority assumption.
- [Angelopoulos & Bates, arXiv:2107.07511v6](https://arxiv.org/html/2107.07511v6):
  finite-sample conformal quantile and exchangeability condition. E05 uses the
  empirical-CDF order statistic rather than NumPy's interpolating quantile
  index convention; if the required rank exceeds n, no finite residual bound
  is available from that calibration sample. Inference consumers must allow
  unbounded intervals; this is not a guarantee for nonexchangeable time series.

Diagnostics propose hypotheses; validation selects configurations. No fixed
ACF cutoff proves a useful lookback, and no stationarity p-value proves forecast skill.

## Verified offline capability and choices

Three development directions were considered: neural architecture expansion,
market-specific features, and a trustworthy small experiment lifecycle. The
lifecycle was selected because current evaluation defects undermine comparisons,
the machine lacks the supported TensorFlow environment, and it delivers a
testable foundation without paid services or large downloads.

`quant_forecast_lab.research` uses the existing CLI/evaluation/provenance layer.
It fits StandardScaler + Ridge on latest-first log-return lags, scores candidate
lookbacks/alphas on common validation origins, freezes the chosen training model,
then evaluates the final chronological test. No refit or test-based epoch choice.
Configurable candidates, raw split boundaries, schema checks and immutable input
snapshot establish the input contract. No technical-indicator/PCA expansion was
introduced without evidence.

`diagnostics` writes ADF/KPSS with their null hypotheses, warnings and settings,
ACF/PACF CSV/plots, readable analysis and z-table reports. These are training-only,
unadjusted pointwise diagnostics; they do not automatically pick lag or establish
causal/predictive significance. Short or constant series are marked inapplicable.

`tracked_run` keeps a unique UTC/UUID run directory, configuration, environment,
source hashes, dirty Git state, closed-log checksums and failed/interrupted status.
JSON numerical model/scaler parameters avoid executable pickle. Reload is checked
against every final-test origin. A future forecast has no invented actual/date.
Run/trial failures retain errors and are not presented as successful evaluations.

Synthetic smoke data is explicitly labelled in manifests, summary and figures.
This pass does not establish market forecasting skill, profitability, calibrated
uncertainty, live execution, walk-forward training in the new runner, or deployment.

The root README/requirements now describe this import instead of the deleted
PyTorch system. Upstream README, author, license and citation were preserved.
The obsolete `vsgod/` ignore entry was removed; targeted text search found no
remaining project branding references in maintained source/docs/configuration.
Instruction text and incidental encoded notebook bytes are not project branding.

## Final local verification

- `python -m pytest -q -o addopts= tests`: 61 tests passed (baseline: 24).
- `../.venv/Scripts/ruff.exe check quant_forecast_lab tests tests_runtime`: passed.
- Formatter check for new modules and the modified CLI/provenance layer: passed.
- `python -m compileall -q quant_forecast_lab benchmarks`: passed.
- Existing CLI help and synthetic `experiment` command: executed successfully.
- `python -m pip check`: no broken requirements in the available environment;
  this does not install or validate missing TensorFlow.
- Root workflow YAML and project TOML: parsed locally. GitHub Actions not run.
- `git diff --check`: passed; original staged diff fingerprint unchanged.
- E2E test prohibits socket connections; final-test mutation leaves fitted
  scaler/model parameters, training correlation table and validation trials
  unchanged. Predictions reproduce saved metrics and artifact hashes verify.

Final demonstration: `log/ridge_log_return/20261111T132459_904057Z_cd6947e4/`.
Status completed; all 26 recorded artifact hashes verified and all 17 package
source hashes match the final code. Synthetic input snapshot SHA-256:
`039f6bfad2c28b11ded7a1848bbfca308540ea0d583a3abede44dc3a7308aaa9`.
Elapsed wall time recorded by the run: 4.48 seconds (one local smoke run, not
a performance benchmark). The package parses under Python 3.11 syntax grammar;
this does not substitute for execution on that interpreter.

## Final independent assessment of this pass

Scores describe the entire current workspace, including unverified legacy paths;
they are not a production certification or a claim that all phases are complete.

| Dimension | Score / 5 | Evidence | Remaining weakness / next step |
| --- | --- | --- | --- |
| Architecture and separation | 3 | Reuses maintained package and CLI; distinct diagnostics/provenance | Legacy scripts still couple downloading, preprocessing and training; migrate only after runtime checks |
| Code quality and maintainability | 3 | Lint clean, validated config, explicit error/state contracts | No comprehensive typing gate; reference runner is intentionally narrow |
| Functional correctness | 2 | Reproduced evaluation defects fixed; offline E2E/reload verified | Full neural runtime and legacy inverse transforms remain unverified |
| ML correctness | 2 | Common origins, train-only scaling/analysis, validation-only selection; future-mutation regression | No new market evidence; legacy findings L01/L02 need resolution |
| Data/experiment reproducibility | 3 | Input snapshot, source/config/data/artifact hashes, seed and version metadata | No complete lockfile or clean supported-runtime reproduction; vendor data can be revised |
| Testing and reliability | 3 | 61 local tests, failed trial/run checks, nonfinite/alignment regressions | Runtime suite blocked; hosted CI and other platforms not exercised |
| E2E completeness | 2 | CSV/synthetic to diagnostics, training, evaluation, report and reload works | New runner has fixed holdout only; acquisition, walk-forward and multi-feature ablation not integrated |
| Deployment readiness | 1 | Offline artifacts and CI configuration available | No measured service SLOs, release, monitoring or rollback; none claimed |
| Documentation and developer experience | 3 | Current Vietnamese quickstart, provenance, limitations and execution record | Upstream historical instructions still require external artifacts and supported environment |
| Practical/portfolio value | 2 | Demonstrable reproducibility and scientific safeguards; genuine engineering contribution | No demonstrated real-data advantage, novel research result or user validation |

Technical verdict: a useful research prototype with verified evaluation repairs
and an auditable offline reference path. ML Engineering verdict: demonstrates
selected competencies in temporal isolation, testing and experiment provenance;
does not yet demonstrate a complete production ML lifecycle. Practical value:
useful for learning and controlled experimentation, market utility unproven.
Production-readiness verdict: **not ready**. Recommendation: **continue with
narrow scope**, keeping simple baselines and resolving legacy correctness first.

## Prioritized roadmap with measurable gates

1. **P0, supported runtime / L01-L02:** obtain a separate Python 3.11/3.12
   environment with TensorFlow 2.18.1; run `tests_runtime` without exclusions.
   Add hand-calculated reconstruction regressions and verify checkpoint policy
   before repairing legacy paths. Gate: all affected runtime tests pass.
2. **P1, Git integration:** resolve the imported-source checkpoint policy with
   the repository owner. Gate: one-file commits can represent verified changes
   without consuming pre-staged user work or creating invalid intermediate trees.
3. **P1, real data:** select an identified, authorized Date/Close snapshot with
   source, adjustment policy and hash. Gate: reproduce a labelled market run
   from that exact snapshot; do not tune against its final test.
4. **P1, walk-forward:** integrate existing fold primitives into the reference
   runner with per-fold train-only preprocessing/diagnostics and common targets.
   Gate: fold isolation tests, no overlapping scored targets unless explicitly
   handled, complete per-fold manifests and aggregate baseline comparisons.
5. **P2, feature contribution:** one justified additional feature group at a
   time, with training-only selection and fixed protocol. Gate: ablation across
   multiple periods/assets reporting all folds/seeds, including negative results.
6. **P2, operational scope:** decide batch-research versus serving needs before
   adding infrastructure. Gate: a clean-environment reproduction plus actual
   hosted CI evidence; deployment claims require their own measured acceptance.

Learner check: explain why an ACF spike is not a validated lookback, why an
observed validation/test history may be used at its forecast origin while future
labels may not select a model, and why a synthetic improvement is not market
evidence. The hand-alignment exercise is in the root README.

## Layout cleanup requested by the owner

The owner explicitly authorized reorganizing the nested project and removing
duplicates, empty items and obsolete files. This supersedes the earlier path
assumption; the move is not an attempt to clean the Git index.

### What was found

- One Git repository, not a nested Git checkout. Package/configuration lived
  one directory below the Git root. Running tests from the root produced 13
  collection errors because `quant_forecast_lab` was not importable there.
  The same suite passed 61 tests from the nested source directory before moving.
- Four name collisions: `.github`, `.gitignore`, `README.md`, `requirements.txt`.
  These were different files/responsibilities, not byte-identical duplicates.
- No zero-byte source files, no empty non-cache directories, and no exact
  duplicate non-log files were found in the imported tree.
- A stale root `__pycache__/main.cpython-313.pyc` referred to removed source.
- Ten historical screenshots had no direct references in code, notebook source
  or maintained documents. The old README merely listed their directory.
- The root VS Code settings selected nonexistent `stock_env` rather than the
  existing `.venv`.

### Keep, merge, relocate, remove

| Item | Decision | Reason |
| --- | --- | --- |
| `quant_forecast_lab/`, tests, benchmarks, examples, root scripts, manifests | MOVE to Git root | One execution and package root; no import-path workaround |
| Existing run logs, datasets, model/scaler fixtures and research evidence | KEEP and move unchanged | Historical provenance and runtime-test dependencies |
| Root README | KEEP and update | Current Vietnamese entry point and verified capability guide |
| Imported README | RELOCATE to `docs/UPSTREAM_GUIDE.md` | Useful inherited LSTM commands and source attribution; adjust relative links and mark unverified claims |
| Two ignore files | MERGE into root `.gitignore` | Preserve secrets/output exclusions; remove redundant rules |
| Dependency declarations | CONSOLIDATE in `pyproject.toml` | Root requirements and Conda now install the same local package with research/dev extras |
| Two CI files | MERGE responsibilities | Root CI retains automatic offline gates; inherited runtime checks retained in manual `legacy-runtime.yml` |
| Benchmark/notebook workflows | MOVE to `.github/workflows/`, manual triggers | Preserve research workflows without starting long training on ordinary changes |
| Issue/PR templates and contributor guide | MOVE and align | Root GitHub configuration; TensorFlow version optional for offline bug reports |
| `_config.yml` | REMOVE | Jekyll presentation theme, no code/build references or active Pages workflow |
| `.github/FUNDING.yml` | REMOVE | Upstream PayPal solicitation; not license/copyright attribution |
| `docs/assets/legacy/` (10 PNGs) | REMOVE | Unreferenced historical screenshots; notebooks and reference/evidence figures remain |
| Old caches | REMOVE | Regenerable bytecode/tool state; fresh caches remain ignored |
| LICENSE, CITATION, source copyright and AGENTS | KEEP | Preserve provenance and working instructions |
| Master prompt and editor paths | UPDATE paths only | Match the single project root and existing environment |

### Safety and verification

Before changing collisions, root configuration/editor files were copied into
the ignored maintenance record. File paths were checked to remain inside the
workspace; no nested Git metadata or reparse points were present. The move
initially verified 191 unchanged file hashes. Later edits are limited to the
documented configuration/documentation changes; Python source, notebooks, model
fixtures and original run logs were preserved.

Maintenance evidence:
`log/maintenance/20261111T142041994Z/before_manifest.json` and
`log/maintenance/20261111T142041994Z/migration_manifest.json`.

After migration:

- `.venv/Scripts/python.exe -m pytest -q -o addopts= tests`: **61 passed** from root.
- `.venv/Scripts/ruff.exe check quant_forecast_lab tests tests_runtime`: passed.
- Package/benchmark compilation and Python 3.11 grammar checks passed.
- Local Markdown destinations in README, contributor guide, docs and benchmark
  reports exist; no broken local link found.
- Four workflow YAML files, Conda configuration and project TOML parsed.
- Notebook root-discovery markers and reference model files remain available.
- Synthetic demo: `log/ridge_log_return/20261111T142503_361580Z_1686091f/`.
- Original staged diff fingerprint remains
  `d6fc3941c3b572ccbcba58ff52a260283ca96069`.

The legacy TensorFlow runtime remains unverified. The inherited runtime workflow
also requires reference fixtures, currently local/ignored, to be provisioned
before a hosted run. Moving YAML does not establish hosted CI success.

No commit was created. Because the original imported paths were already staged,
Git now shows old paths deleted in the worktree and new root paths untracked.
This accurately reflects a filesystem move while preserving the user's index.
The cross-file relocation still conflicts with valid one-file intermediate
commits until the owner resolves the import/commit policy.

## Owner-approved bootstrap publication

After reviewing the conflict in AGENTS.md section 8.2, the owner explicitly
approved one exception: commit the complete imported/refactored project in one
atomic bootstrap commit, then push normally to `origin/main`. This authorization
also covers replacing obsolete staged nested paths with the reviewed root layout.
It does not change the one-file policy for later development.

Publication preflight: current Git identity matches the latest repository-owner
commit; the remote `main` tip was verified as
`2557a568829339f3ea1db3cef563e11d1ce67f5c`, matching the local base. The intended
tree has 80 files (about 1.98 MB); generated logs, environments and model binaries
are excluded. Checks of private-key, GitHub-token and AWS-access-key patterns
reported no matches in the intended files; this is a scoped check, not a claim
of comprehensive security review. Final verification returned 61 passing tests,
clean Ruff results and successful compilation. The TensorFlow runtime limitation
remains documented and is not represented as a passing check.

Planned message: `feat(research): add reproducible forecasting pipeline`.
The staged-tree whitespace check exposed inherited trailing spaces/end-of-file
blank lines not visible in the earlier unstaged-only check. These were normalized;
Python ASTs were compared before/after and remained identical. Markdown hard
line breaks in AGENTS.md were preserved using backslashes; no policy was changed.
The resulting commit hash and verified remote outcome are recorded in the local
publication receipt and the handoff message after the push; no force-push or
history rewriting is authorized.
