# Engineering review and execution record

Current layout (2026-11-11): the imported project has been flattened into
`D:/Github/StockForecasting`. Run commands from this Git root. Earlier commands
and directory references below describe the pre-migration state and are retained
as historical evidence. See the layout-cleanup record at the end of this file.

Review date: 2026-11-11. Scope: imported Quant Forecast Lab source inside
`D:/Github/StockForecasting`; parent Git branch `main`, HEAD `2557a56`.

## Current execution: main integration and branch cleanup (2026-11-23)

The default branch now contains the verified pipeline. [Pull Request #5](https://github.com/TranMaiQuynh/stock-forecasting/pull/5)
passed its hosted checks and GitHub recorded it as merged at `20b74a6` after a
fast-forward of `main` from `1e6769b`. All 152 incoming commits were checked:
exactly one changed file each, owner author identity only, no merge commits.
No existing commit was squashed, rebased, amended or rewritten.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | P1 | Default branch | Public main still exposed the old implementation | Review PR, require passing PR CI, fast-forward and push main | PR #5 merged; run 45066904389 passed; main run 45067159605 passed | DONE | Existing `20b74a6` preserved |
| M02 | P2 | Branch cleanup | Two completed remote fix branches obscured the canonical version | Verify both tips are ancestors of main, remove remote refs and merged local fix ref | GitHub API and ls-remote show only main; unmerged local backup preserved | DONE | Ref cleanup; no content commit |
| M03 | P2 | Reviewer onboarding | Badge and clone command still targeted the deleted development branch | Point README to main; independently clone main, install wheel and run demo | README syntax passed; run 45067295141 passed; fresh main clone verified 208 demo artifacts and rollback/restore | DONE | `b6af9cb`; this review |

Canonical portfolio URL: [stock-forecasting](https://github.com/TranMaiQuynh/stock-forecasting).
Hosted main evidence: [run 45067295141](https://github.com/TranMaiQuynh/stock-forecasting/actions/runs/45067295141)
for README commit `b6af9cb`. Runtime source, tests, workflow and lockfile remain
identical to the release with 262 installed Windows tests and successful PR CI.

The new `log/validation_main_20261123/main-clone` was cloned directly from
GitHub main. It used a new `.venv/replay-demo` and non-editable installation
from `uv.lock`, reusing the previously downloaded dependency cache offline.
No private fixtures or pilot artifacts were supplied. Its installed demo
returned exit 0, 2 forecasts/2 labels, 21 completed child runs and 208 verified
artifacts, with rollback/restore verified. Expected historical-incident health
remains unhealthy and acceptance not_ready; synthetic evidence is not alpha.
Receipts, ancestry checks and complete stage logs are under
`log/validation_main_20261123/`. The final documentation-only commit receives
its own hosted main check; its result is retained in the local handoff receipt.

GitHub now has one branch, `main`. Local `backup-local-work-before-pull` at
`c5b9512` contains unmerged owner work and is intentionally retained, never
published or deleted. The untracked owner note is also preserved. No market
training, live-data modification, scheduler change or infrastructure deployment
was performed. Production and predictive-performance limitations remain as
documented below; this task improves publication and discoverability.

## Previous execution: GitHub and clean-clone handoff (2026-11-23)

Scope: publish the already verified release, verify hosted CI, and reproduce the
lightweight installed demo from GitHub. No model search, market-candidate
training, live-ledger mutation, scheduler changes or merge into `main` performed.
The owner note remains untracked and untouched. Author identity was checked;
every published commit since remote `b7dff83` changes exactly one file.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | P1 | Publishing/CI | Latest verified release was local only | Push development branch; inspect actual hosted workflow results | Commit `01ae535`, hosted run 45063351497: installation, installed import, lint/compile, offline tests, demos and artifact upload all succeeded | DONE | `01ae535` published |
| C02 | P1 | Reproducibility | Reviewer must not depend on private pilot files | Clone branch directly from GitHub; install non-editable wheel using lockfile and empty dependency cache into a new environment | 262 tests passed without skips; synthetic batch demo verified 208 artifacts, rollback and restore; checkout remained clean | DONE | `01ae535` verified |
| C03 | P2 | Onboarding | README lacked the current branch quickstart and CI evidence | Add clone/install/test/demo commands, branch-specific badge and expected incident outcomes | PowerShell and bash syntax checks; commands match verified installed demo; no runtime source changes | DONE | `867faaa`; this review |

Hosted evidence: [release run 45063351497](https://github.com/TranMaiQuynh/stock-forecasting/actions/runs/45063351497).
At that handoff the development version was on `fix/forecasting-validation`
and `main` remained unchanged. The historical snapshot is preserved at
[commit 20b74a6](https://github.com/TranMaiQuynh/stock-forecasting/tree/20b74a6c81cdbd1c12706ef3d7d0a314458e3aeb).
The main integration above supersedes the earlier branch-specific portfolio link.

Independent Windows checkout at `log/validation_cv_20261123/clean-clone`
contains only GitHub-tracked source initially; no pilot data, model or fixture
was copied into it. Python 3.12.16 was reused as the base interpreter, while its
virtual environment and dependency cache were newly created. `uv 0.13.7 sync
--locked --extra research --extra dev --no-editable` downloaded dependencies;
the installed package resides in the new environment's `site-packages`, with
no TensorFlow. This is a clean checkout/environment check on the existing host,
complemented by the separate hosted Linux runner, not a new Windows host test.

Windows result: **262 passed, 1113 warnings in 424.86 seconds**, no skips.
Lint and compile passed. The demo returned exit 0, 2 forecasts/2 labels, verified
rollback/restore and 208 artifacts across 21 completed child runs, retaining
the deliberately failed child run. `health_status=unhealthy`, acceptance
`not_ready` and streak 1/2 are expected: historical lateness survives recovery.
Synthetic fixtures are not market evidence; promotion remains disabled.

Local receipts and complete output: `log/validation_cv_20261123/`, especially
`clean_receipt.json`, `clean_tests.txt`, `clean_demo_retry.txt` and
`clean_demo_verification.txt`. The first verification driver mistakenly invoked
the script with `-I`, hiding its sibling `demo_shadow` import. That harness
failure and initial receipt are retained; the documented script command without
`-I` passed, followed by a separate installed artifact-verification command.
No project implementation or test was weakened to address the driver error.

Assessment: the repository now has public CI evidence and a demonstrated demo
path suitable for a CV. The prior production-readiness and predictive-skill
limits below still apply. The 20 neural-runtime tests remain separately verified
historical local evidence; this lightweight clean-clone run does not run them
or require their private reference fixtures. macOS remains unverified. Hosted
workflow artifacts have seven-day retention, so durable reproduction relies on
the committed source, lockfile and documented commands rather than that upload.

## Previous execution: first-week operational review (2026-11-23)

Starting HEAD `f238a7d`, branch `fix/forecasting-validation`; tracked tree/index
clean, owner author identity unchanged, untracked owner note preserved. Read-only
audit at 2026-11-23 local time found 8 live forecasts and 6 labels: each asset
missed origin 2026-11-16, delivered origins 18/20 after the configured deadline,
and labelled origins 17/19 late. Origin 20 labels are not due yet. Eight jobs
succeeded and no failed stage events exist; execution gaps are verified, but
host power/login/battery cause is unverified. The health task's exit 1 correctly
signals the missing origins, not an application crash. No claim of 20-session
acceptance, predictive edge or production readiness follows from these data.

Baseline: installed v2 **239 offline tests passed, 729 warnings**, no skips;
`log/validation_20261123/baseline_offline.txt`. Read-only audit receipt:
`pilot_before.json`. KEEP frozen models/protocols and all existing records.
FIX loss of historical label-timeliness visibility. EXTEND the existing CLI
with an explicit operational acceptance report. VERIFY coordinated backup and
restore using real ledger/state/artifacts. DEFER paid/remote infrastructure and
new model searches; a powered, logged-in desktop is not a continuous host.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | P1 | Acceptance | Successful jobs do not establish 20 timely sessions | Read-only per-session paired delivery/label audit with consecutive-window gate | 15 unit cases; integrated CLI/artifact checks; no ledger mutation; live rule remains 20 | DONE | `c60a920`, `291b641`, `20f564e`, `4bd278d` |
| S02 | P1 | Health | Late labels disappear from pending alerts after recovery | Retain historical late forecasts/labels in version-2 independent report | 42 focused tests; 2 transition checks on isolated v1; recovered lateness remains visible | DONE | `6dcf82f`, `8728844`, `d975f00` |
| S03 | P1 | Recovery | Only individual SQLite restore was demonstrated | Pause owned tasks; coordinated DB/artifact inventory and restore drill into new paths | 3 failure/recovery tests; actual 8 predictions/6 labels and 2 matching copied-inference probes; tasks resumed | DONE | `8b3f705`, `e926e0c` |
| S04 | P1 | Release/handoff | Installed runtime must match verified source | Isolated release, installed tests/demo, task transition, learner explanations and updated review | 262 installed offline + 20 neural-runtime tests; source/wheel pin and actual v3 task results | DONE | `45b5043`; this review |
| O07 | P1 | Future pilot acceptance | First observed week failed timing criteria | Continue frozen collection; require 20 consecutive future compliant sessions | Actual recorded evidence only; no retrospective forecasts or rewritten history | TODO | future operation |

### Verified handoff and limits

Final installed `.venv/batch312-v3` regression: **262 passed, 1113 warnings in
439.47 seconds**. Separate existing TensorFlow runtime: **20 passed, 8 warnings
in 53.03 seconds**. All **282 current tests** passed across these environments;
no tests were skipped or weakened. Ruff, compileall, locked dependency check and
55-package compatibility check passed. Warnings remain recorded, primarily
pandas timezone deprecations. Receipts/logs are under `log/validation_20261123/`.
The first focused run exposed an old v1 expectation that a recovered late
forecast clears health; its failed log is retained. The strengthened test
supports v1/v2 report contracts explicitly. An isolated source export with the
old health implementation passed both transition tests before the one-file
commits changed the health report to v2.

Installed source/wheel pin:
`00a28732affc838fdb03b12b532634ccc9fc0ca9042dbb6aec8e731534ec65e1`.
Active deployment `equity-pilot-v3` uses
`log/operations/live/config-release-v3.json`; both existing task actions use
`.venv/batch312-v3`, preserving their wrapper/schedule and the v2 runtime for
rollback. Models, protocols, training history and existing forecast/label rows
are unchanged. No market-candidate fit, search or holdout analysis was performed;
tests and demos did train their explicitly synthetic fixtures.

Actual Task Scheduler receipt `final_scheduler.json` confirms owner `aliza`,
Interactive mode, ready tasks, 15/5-minute intervals and expected results:
**run=0, health=1**. The health result is a deliberate incident signal, not a
crash. Battery restrictions and lack of wake-to-run are verified; these settings
were preserved. They are possible availability constraints, not proof of what
caused each past gap. No whole-host uptime claim is made.

Actual `live_acceptance.json`: **not_ready**, **0 consecutive compliant
sessions**, assessed origins **2026-11-16 through 2026-11-20**; 18 model/input/
inference runs verified, no current runtime blockers. Every origin has at least
one missing/late forecast or late label for the paired assets. The 2 pending
labels are still future observations, not failures. Historical incidents are
retained; a later 20-session window is a separate review, not an erasure of
those incidents. Replay cannot supply prospective evidence. The only 3 labelled
observations per asset currently give descriptive RMSE skill versus zero of
approximately **-6.82% AAPL / -1.58% MSFT**. This tiny, timing-impaired sample
neither establishes alpha nor justifies retuning the frozen candidates.

The installed synthetic demo
`log/batch_demo/20261122T191135_187897Z_29b6bfad` verified **208 artifacts** and
21 completed child runs, retained its injected failed run, and deliberately
returns `health=unhealthy`, `operational_acceptance=not_ready`, streak 1 of 2.
Its database is recovered, but its historical late delivery is not made timely.
The demo stays explicitly synthetic/replay, with promotion disabled.

Coordinated real-data capture paused both owned tasks and resumed them in
`finally`. Backup:
`log/recovery_20261123/pilot_backup/20261122T191322_297347Z_c8cb8cfb`:
**26 verified source directories / 271 inventoried artifacts**, including both
logical databases, protocols, forecast/label input sources and active v2 code.
Restore drill:
`log/recovery_20261123/pilot_recovery_drill/20261122T191333_572703Z_b1b7dec2`:
**254 artifacts**, matching 8 forecast and 6 label records, operations table
contents and 2 offline numerical forecasts from copied sources. Database
`data_version` guards reject external commits during capture. No real ledger
was overwritten and no replay probe was recorded as a prospective forecast.

This is local serving recovery. Absolute ledger paths remain; relocated live
activation, dependency-wheel provisioning, task recreation, offsite storage,
continuous-host availability and full-host failover are unverified. Earlier
unverified container status is not reclassified as complete. New hosted CI has
not run because no push was requested. Source, tests and demo are ready for an
authorized push; the existing owner note stays untracked.

### Independent assessment after first-week evidence

| Dimension | Score / 5 | Evidence | Remaining weakness / measurable next step |
| --- | --- | --- | --- |
| Architecture | 3 | Separate research, immutable ledger, jobs, health, acceptance and recovery tool | Internal shadow helpers remain shared; extract public contracts only when justified |
| Code quality | 3 | Bounded retries, input validation, explicit versioned health contract and Ruff | Static typing is partial; strengthen high-risk interfaces incrementally |
| Functional correctness | 4 | 282 passing tests, partial-commit, corruption, concurrent capture and real restore checks | Actual delivery deadlines failed; require 20 consecutive observed compliant sessions |
| ML correctness | 4 | Frozen checkpoints/protocols, temporal tests and matched zero-return baseline | Only 3 labels per asset; do not promote or tune on this evidence |
| Reproducibility | 4 | Locked wheel install, source parity, lineage and copied offline inference | Vendor revisions/PIT quality and data licensing remain unresolved |
| Testing/reliability | 4 | Installed offline + neural regressions, synthetic incidents and actual recovery | Run revised hosted Linux CI after authorized push; measure longer-term resource behavior |
| E2E completeness | 3 | Actual acquisition/forecast/label records plus scheduled health and acceptance audit | Future operational acceptance and full-host recovery remain incomplete; no automatic promotion |
| Deployment/operations | 2 | Real owner tasks, isolated v3 release, retained rollback runtime and actual backup drill | Desktop gaps/battery/login remain; provision continuous host and external outage detection |
| Documentation/DX | 4 | Vietnamese walkthrough, real incident explanations and runnable commands | Follow offline demo before live; market candidates/receipts are local artifacts |
| Practical/portfolio value | 3 | Real auditable forecasts, explicit SLO failure and verified recovery | No predictive advantage, investment benefit, production uptime or adoption evidence |

**Technical verdict:** maintainable research and local prospective batch system
with tested operational controls. **ML Engineering verdict:** credible lifecycle
and reliability evidence; model superiority is unproven. **Practical value:**
usable to collect, trace and audit forecasts and study operational failures.
**Production readiness:** not ready for unattended production. **Development
recommendation:** continue the narrow frozen pilot, prioritizing availability
and repeatable recovery over model complexity.

Roadmap: (P1) ensure power/login/network or provision an authorized continuously
available host and independent watchdog; (P1) demonstrate 20 consecutive timely
paired forecast/label sessions under the frozen criteria; (P1) implement offsite
backup and rehearse full restoration/activation on a second host with measured
RTO/RPO; (P1) establish permitted point-in-time data use; (P2) run hosted CI and
set retention from measured storage needs. A compliant operational window still
requires separate, adequately powered prospective predictive validation before
any model-promotion or profitability claim.

## Previous execution: scheduled batch operations (2026-11-16)

The owner authorized the next roadmap after `19fda3d`. Branch remains
`fix/forecasting-validation`; tracked tree/index were clean, owner identity
unchanged, and the untracked Vietnamese note is excluded. Existing research
results/protocols and candidate checkpoints remain frozen. No new search,
long training, trading or public deployment is part of this increment.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O01 | P1 | Job state | Individual CLI commands have no durable workflow state or lease | Transactional per-origin jobs, stage checkpoints and bounded ownership | Concurrency, crash/reclaim and checkpoint preservation tests | DONE | `3cd1ab5`, `d277e4e` |
| O02 | P1 | Daily batch | Acquisition/scoring/recording need manual coordination | Calendar-aware runner; transient-only retries; independent per-asset failures | Offline replay, retry/nonretry, stale data, no-fit and crash-resume checks | DONE | `b845fa9`, `14d35e9`, `15fa599`, `4d4f007` |
| O03 | P1 | Monitoring | An absent job cannot report its own failure | Independent due-session/heartbeat/label checker and local human-readable report | Never-started jobs, gaps, concurrent publishers, stale leases and corrupt DB tests | DONE | `4424dd5`, `c434010`, `fd155d9`, `4aeae31` |
| O04 | P1 | Release/recovery | Operational config changes and DB recovery are manual | Pinned deployment activation/rollback and consistent SQLite backup to a new destination | Invalid activation preserves active version; rollback/backup recovery drill | DONE | `9b6fa7f`, `92cfafc`, `c9802df`, `6ec4c9b` |
| O05 | P2 | Runtime/deployment | Default dependencies include neural runtime; no scheduling entry point | Lightweight locked installation, explicit neural extra and Windows scheduler | Installed wheel/source parity, no TF, dependency check, real task exit codes | DONE | `af471bc`, `2d9d779`, `32b7d5f`, `9edacb9`, `6a375d9`, `5bd3c54` |
| O06 | P1 | Validation/pilot setup | Future observations cannot be simulated as evidence | Regression, synthetic incident demo and real preflight; install local pilot | Retain failed runs; no fabricated live forecasts; honest review and handoff | DONE | `4247deb`, `6c1b569`; this review |
| O07 | P1 | Prospective acceptance | Future market sessions have not occurred | Observe 20 consecutive eligible sessions; record incident/recovery evidence | Timely unique forecasts, timely labels, valid hashes and no unresolved failures | TODO | future operation |

KEEP the existing ledger, contracts, acquisition and inference APIs. Add a small
SQLite operations store beside the forecast ledger; do not introduce a workflow
service. Individual snapshot/model/forecast runs keep their current inventories.
Mutable heartbeat/job state belongs outside those immutable run directories.
Scheduler polling uses exchange-session due times instead of fixed UTC close
assumptions. Docker client exists but its daemon is unavailable even outside the
sandbox; a container configuration cannot be reported as runtime-verified here.

### Batch implementation evidence and final independent review

The original baseline this increment was **228 passed, 237 warnings**. A full
intermediate run passed **257 tests, 725 warnings**; a concurrent-publisher test
was added afterward. The locked non-editable environment `.venv/batch312`
passed **238 offline tests, 717 warnings**. Existing full TensorFlow environment
`.venv/runtime312` independently passed **20 runtime tests, 8 warnings** in
24.21 seconds. No tests were skipped or weakened. Logs are retained in
`log/validation_20261116/batch_*`.

Independent review reproduced a stale-green defect: corrupting the forecast DB
made the health command fail while the HTML/JSON retained its old status.
`health_corruption_before.txt` records the failing regression; `fd155d9` makes
downstream monitoring failures publish unhealthy output. The expanded focused
suite passed **25 tests, 488 warnings**. Report publication uses unique temporary
files and bounded retries for Windows sharing conflicts; JSON and HTML are
individually atomic, not a transactional pair.

The installed-package regression then caught a second issue: uv's default local
build cache reused an older wheel when only Python files changed. Cache keys now
include package sources, metadata and license; the installed demo rejects a
source/wheel mismatch. The rebuilt separate `.venv/batch312-v2` passes the corrupt
DB/concurrent report tests, with 55 compatible packages and no TensorFlow. The
previous runtime is retained. Installed source pin is
`a4e271a016fee73e3f495576fdecc3c94615dbbaef85bfa1bb0d207fe81923e0`.
Final installed v2 regression: **239 passed, 717 warnings in 286.30 seconds**;
`batch_final_installed_tests.txt` is the complete output. Together with the 20
runtime tests, all **259 current tests** passed across the two environments.
Ruff, compileall, `uv lock --check` and `uv pip check` passed. Warnings are
timezone deprecations in pandas/plotting and scalar conversion in legacy Keras;
no warnings were hidden to make tests pass. An unactivated stale-build config
and failed test logs are retained as evidence, not used as the active release.

The synthetic incident demo exercises missing jobs, transient retries, failure
after business commit, resume without duplicates, final-label scoring after the
forecast budget, operational rollback and DB restore. Its completed receipt has
2 forecasts/2 labels, `market_evidence=false`, 21 verified completed child runs
and 207 inventoried artifacts. The injected failed run remains failed and its
artifact checksums are verified. These are execution checks, not market results.

Windows tasks `StockForecasting-Pilot-Run` (15 minutes) and
`StockForecasting-Pilot-Health` (5 minutes) were registered under owner account
`aliza`, Interactive/Limited, using versioned hidden wrappers and a working
directory outside the source root. Both executed with exit 0, not merely
registered. Scheduler receipts and stdout/stderr are retained locally. Before
the first origin, health is `not_due` and both assets have zero forecasts/labels.
First eligibility is 2026-11-16 21:30 UTC, subject to data arrival and polling.

Final active release is `equity-pilot-v2`, configured in
`log/operations/live/config-release-v2.json`; both task actions now use
`.venv/batch312-v2`. The transition paused scheduling, verified no tasks were
running, activated the pinned definition, changed both actions and resumed them.
The previous `equity-pilot-v1` definition and `.venv/batch312` remain available;
this is an operational release change, not a model retrain or ML promotion.
`scheduler_verification_v2.json` records the final account/actions/exit results.
The final installed synthetic demo is
`log/batch_demo/20261116T094904_145022Z_f9e75a65`; its completed top-level
manifest was independently reverified (207 artifacts).

The operational pilot uses frozen candidates prepared earlier today, not a
confirmation of the v2 monthly-refit research method. No model search, new
performance evaluation, holdout tuning, long training, trading or public
deployment occurred. No Whisper components apply. Branding search found only a
historical audit reference and incidental encoded image bytes; notebook sources
contain no branding references. Licenses and attribution remain intact.

Current scores are scoped engineering judgments, not certifications:

| Dimension | Score / 5 | Verified evidence | Remaining weakness and next acceptance |
| --- | --- | --- | --- |
| Architecture | 3 | Research, immutable ledger, operations state and watchdog have separate responsibilities | Some internal shadow helpers are reused; expose public contracts only when another consumer needs them |
| Code quality | 3 | Typed exceptions, validation, bounded retries, small modules and passing Ruff | Interfaces are not comprehensively statically typed; add types around high-risk boundaries incrementally |
| Functional correctness | 4 | Full regression plus injected partial-commit, concurrency, corruption and recovery cases | Actual daily vendor/host failures remain unobserved; pass O07 |
| ML correctness | 4 | Temporal isolation, fixed protocols, paired baseline and no-fit inference tests | Predictive edge absent; no promotion without an adequately powered prospective evaluation |
| Reproducibility | 4 | Locked dependencies, source/wheel pin, artifact inventories and self-contained synthetic demo | Market data can be revised; obtain point-in-time licensed data before stronger claims |
| Testing/reliability | 4 | Offline, runtime and incident checks; failed runs retained | New hosted Linux workflow has not run; execute it after an authorized push |
| E2E completeness | 3 | Data → selected checkpoint → scheduled inference → delayed label → monitoring → recovery | Training selection remains a deliberate research operation; no automatic retraining/promotion or full-host recovery |
| Deployment/operations | 2 | Actual local Windows tasks and isolated installed runtime | Login/host availability required; add external watchdog, continuous host, offsite backup and recovery drill |
| Documentation/DX | 4 | Vietnamese learning/runbook explanations, architecture and reproducible commands | Clean clone still needs market candidates; follow synthetic demo first |
| Practical/portfolio value | 3 | Useful prospective forecast audit trail and reliability demonstration | Useful research/engineering tool; unproven investment benefit and no user/deployment adoption evidence |

**Technical verdict:** a defensible research and local batch pilot, materially
stronger than a training notebook. **ML Engineering verdict:** credible evidence
of lifecycle, evaluation and reliability engineering; no evidence of alpha.
**Practical value:** usable for collecting auditable prospective forecasts and
studying failure/recovery. **Production readiness:** not ready for an unattended
production service. **Recommendation:** continue with a narrow operational
pilot; stop adding model complexity until a predeclared experiment justifies it.

Priority roadmap: (P1) pass O07's 20-session window without unresolved delivery
failures; (P1) exercise coordinated two-DB/artifact recovery on a continuously
available host with offsite backup and an external outage check; (P1) resolve
data licensing and point-in-time limitations; (P2) run the revised hosted CI and
measure resource/retention needs. Freeze prospective analysis and sample-size
rules before any claim of predictive improvement. The 252-origin budget and 1%
practical threshold remain unpowered illustrative rules. No production,
profitability or automatic model-promotion claim is authorized by these results.
See [OPERATIONS.md](OPERATIONS.md) for operational and learner instructions.

## Previous execution: shadow pilot (2026-11-16)

The owner authorized the prospective forecasting work discussed after v2. Starting
HEAD: `b7dff83`, branch `fix/forecasting-validation`; tracked index/worktree clean.
The untracked Vietnamese development note remains owner work. Git identity is
`TranMaiQuynh <tranmaiquynhds@gmail.com>`. No expansion of the v2 search or
re-scoring/tuning on the inspected 2023-2025 final holdout is performed. A later
candidate may train on labels already observed by its actual preparation time;
those labels are no longer an untouched evaluation set.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | P0 | Forecast contract | Observation dates do not identify availability or the next exchange session | Validate exchange sessions, data receipts, forecast deadlines and explicit replay/live modes | 15 session-contract tests: holidays, DST, early closes, gaps, stale/future data and incompatible label basis | DONE | `ea83559`, `df0820f` |
| P02 | P0 | Prospective record | Unique run directories do not prevent duplicate or overwritten business forecasts | Transactional append-only SQLite forecast/label ledger; first prediction/label retained; fixed budget | 12 ledger tests: retries, concurrency, conflicts, actual live clock, matched errors and budget | DONE | `ebb6793`, `95d9ba5`, `9e1234e`, `9d99c76`, `a99e42d` |
| P03 | P1 | Batch integration | Saved inference has no prospective record/score workflow | CLI capture/prepare/init/record/score/report; frozen checkpoint and protocol | 16 integration tests including CLI, offline/no-fit inference, stop decisions, gaps, source protection and failed fits | DONE | `a77169a`, `02ef785`, `73ec6c2`, `82f43eb`, `7b32718` |
| P04 | P1 | Retraining/stop decision | Retraining without a frozen rule cannot establish improvement | One refit of pinned recipe; separate fixed-checkpoint operational pilot from v2 method confirmation | sklearn coefficient parity; 2 real market candidates fit through 2026-11-13; no new search, old-final metrics or live forecasts | DONE | `a79ddca`; local preparation receipt below |
| P05 | P2 | Handoff | Operational behavior and limits need a reproducible demonstration | Offline demo, learner explanations and independent review | 228 tests, lint/compile; clean source export demo verifies 111 artifacts; one-file commits | DONE | `c8d7424`, `fb82a2a`, `217735e`, `19ae73d`, `892589b`; this review |
| P06 | P1 | Live operations | No scheduled daily pilot or prospective labels yet | Configure acquisition/record/score schedule with heartbeat, recovery and an explicit operational acceptance window | Proposed 20 consecutive valid sessions, timely receipts, no duplicate predictions, failure drills; not statistical confirmation | TODO | not deployed |

Decision: KEEP the research/selection/artifact components; EXTEND with a narrow
local batch lane. SQLite provides atomic uniqueness without a new service;
exchange-calendars 4.13.2 is already declared and installed. Calendar checks do
not prove vendor point-in-time correctness. Live receipts use the machine clock;
historical replay is explicitly labelled and cannot count as prospective proof.
Installed exchange-calendars 4.13.2 resolves XNAS to the XNYS calendar; this is
a provider calendar alias, not independent Nasdaq venue/calendar verification.
No external timestamp authority, data-license clearance, scheduler deployment,
model promotion, trading, profitability or future observations are claimed.

### Verified evidence and scope

Baseline: all 185 pre-existing tests passed before implementation. Final local
regression: **228 passed, 237 warnings**, no skipped/excluded tests, Windows
Python 3.12.16 with the existing runtime. The extra 43 tests cover the new lane.
`log/validation_20261116/full_tests.txt` retains the output. Warnings are from
pandas/Keras and exchange-calendars' pandas-3.1 timezone access; none are hidden.
Ruff passed for maintained source/tests/runtime tests/scripts and both research
benchmark runners; compileall and `git diff --check` passed.

Synthetic demo `log/shadow_demo/20261116T043803_369892Z_90951d29/` ran one fixed
refit, two forecasts and two delayed labels; retry created no duplicate. All 10
child runs and 111 total artifacts verify. Its negative descriptive skill causes
`stop_candidate_retain_baseline`, as required. Two synthetic observations have
no statistical/market interpretation. A clean tracked-source ZIP export of
`217735e` repeated the same complete demo with the same numerical result:
`log/validation_20261116/clean_export_demo/shadow_demo/20261116T060759_364221Z_b40b83de/`.
This export reused the supported installed environment; it is **not** a new
dependency installation or a full independent-environment replay. CI was extended
to run this demo; no new hosted CI run is claimed without a push.

Yahoo acquisition initially failed inside the sandbox and succeeded on the
authorized retry outside it; failed manifests were retained. Each new adjusted
close snapshot contains 2,985 sessions from 2015 through 2026-11-13. Both pass
continuous-session validation. Original v2 root still verifies 5,132 artifacts.
The source recipe is taken from the **last**, not best, development fold of each
already-selected policy. No candidate search or market performance evaluation
was conducted on the new snapshots.

| Asset | Pinned lookback/lambda | Eligible new fit labels | Fit cutoff | Candidate run |
| --- | --- | --- | --- | --- |
| AAPL | 10 / 10, alpha=29,740 | 2,974, expanding | 2026-11-13 | `log/shadow_model/20261116T060756_908555Z_1212f86b` |
| MSFT | 10 / 10, alpha=7,560 | 756, rolling history | 2026-11-13 | `log/shadow_model/20261116T060758_522041Z_653cc419` |

Full hashes/source/snapshot/protocol references and replayable preparation script
are local under `log/validation_20261116/market_preparation.json` and
`prepare_market_candidates.py`. These data/artifacts are ignored, not redistributed.
The current untracked owner note and documentation edits are disclosed by the
run's dirty-worktree flag; preparation is not claimed as a clean benchmark.
Protocols under `log/shadow/protocols/` pin each candidate, earliest origin
2026-11-16, 252-forecast upper budget and 1% descriptive RMSE threshold. These
values are pilot assumptions, **not** a power analysis or evidence of usefulness.
The proposed 20-session operational review is separate and need not wait for
252 labels; neither count establishes predictive skill.

Both real candidates passed saved-model reload and artifact verification. The
pre-start recording attempt was correctly rejected because snapshot origin
2026-11-13 precedes the allowed origin; **zero live forecasts/labels exist** and
no market ledger was created. With the default buffer, the first eligible
2026-11-16-origin receipt is after 21:20 UTC (04:20 Vietnam, 2026-11-17), provided
the actual source close is available, and before the next session opens.

**Important boundary:** both serving checkpoints are frozen for this pilot.
The MSFT recipe inherits the 756-label fit scope, but monthly retraining and
reselection are not implemented in the live pilot. This is not an independent
confirmation of the complete v2 research policy. V2's old results are unchanged.

### Independent reassessment (0-5)

| Dimension | Score | Evidence | Remaining weakness / next gate |
| --- | --- | --- | --- |
| Architecture | 3 | Existing research/inference retained; session, ledger, refit and CLI responsibilities separated | Legacy neural paths remain coupled; keep operational scope small |
| Code quality | 3 | Explicit mode, frozen protocol, transactional uniqueness and validation | No complete static typing gate; review broader legacy interfaces |
| Functional correctness | 3 | Existing 185 tests plus 43 regressions pass; actual market refits reload | No long-running scheduler/failure-recovery evidence |
| ML correctness | 3 | Pinned recipes, causal labels, baseline comparison and stop logic | No new predictive advantage; separate complete v2 confirmation required |
| Reproducibility | 3 | Source export demo, snapshot/model hashes and fixed seeds | Market data is local, vendor revisions and cross-platform lock remain unresolved |
| Testing/reliability | 3 | 228 tests, concurrency/clock/solver failures and no-fit inference checks | New hosted Linux run and real multi-session failures not yet measured |
| E2E pipeline | 3 | Snapshot to optional refit, recorded prediction, delayed label and report works in replay | Live collection, scheduling and registry/promotion remain pending |
| Deployment | 1 | Usable local batch CLI and CI configuration | No deployed scheduler, heartbeat, measured SLO or rollback drill |
| Documentation/DX | 3 | One-command public-data-free demo; Vietnamese timeline and live commands | Fresh Linux installation and long-term onboarding still need verification |
| Practical/portfolio value | 2 | Auditable new engineering contribution and measured negative demo behavior | No user pilot, confirmed market benefit or profitability |

Verdicts: technically defensible research and tested local batch software;
credible ML Engineering work with bounded claims; practical market/user value
unproven; production **not ready**. Continue the narrow scope. Next gates are
scheduled live acquisition/recording with heartbeat and incident drills, a
20-session operational review, and a separately registered statistically justified
confirmation design if predictive quality remains worth investigating. Stop weak
candidates rather than extending budgets based on observed outcomes. Do not add
trading, neural architectures or infrastructure to conceal absent utility.

## Previous execution (2026-11-14)

### Baseline-eligible adaptive forecasting

The owner authorized implementing the reviewed research improvements. Starting
HEAD is `9126fc5`; the existing untracked Vietnamese development note is user
work and is excluded from edits/commits. Baseline verification: **142 passed,
24 warnings**, supported Windows runtime, no exclusions; evidence is
`log/validation_20261114/baseline.txt`.

| Task ID | Priority | Component | Issue | Action | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V01 | P0 | Research protocol | Previously inspected holdout cannot confirm new changes | Register bounded development-only v2, exclude 2023 onward; reserve future confirmation | Protocol committed before execution; exactly 704 candidates; no acquisition | DONE | `b88cdb2`, `57afaf2` |
| V02 | P1 | Selection | Ridge must win even when baselines are better; penalties narrowly scaled | Add baseline-eligible validation and alpha=n*lambda through existing runner | Hand-calculated baselines, normalized penalties, deterministic ties and failure checks pass | DONE | `4143f3f`, `f9cc946` |
| V03 | P1 | Model updates | Fixed model can age; refits need explicit information boundaries | Implement frozen/monthly and expanding/rolling scopes with every numerical checkpoint retained | Future mutation, observed-origin schedule, origin availability and per-origin reload checks pass | DONE | `f9cc946`, `39447d1`, `4ed7a93` |
| V04 | P1 | Development evaluation | New method needs fair baseline comparisons and uncertainty | Reuse walk-forward orchestration with development-only mode; paired block intervals; bounded market replay | All 8 comparisons/704 candidates; 73 clean manifests, 3,160 fit states and 8,064 reloads independently checked | DONE | `7597cfb` |
| V05 | P2 | Handoff | New contracts need an understandable, reproducible entry point | CLI, saved prediction compatibility, learner explanation and measured report | Clean checkout/independent venv: 185 tests, 14 stages; hosted Linux offline CI succeeded at 336d9c3; final publication receipt records the documentation SHA | DONE | `336d9c3`, `fc80481` |
| V06 | P1 | Independent confirmation | Future predictions and their labels do not yet exist | Freeze method and prospectively record predictions with origin no earlier than 2026-11-16, then evaluate delayed labels | Protocol registered; no future forecasts or independent confirmation claimed | BLOCKED | future observations required |

KEEP existing preprocessing, diagnostics, numerical artifacts and logging.
EXTEND the existing research/walk-forward runners; preserve their default v1
behavior. DEFER new features, neural models, volatility targets and execution
until these bounded comparisons establish whether the basic changes help.
Production operation and profitability remain unverified; prospective labels
after the new protocol cannot be manufactured or replaced with old test data.

Measured run: `log/baseline_guard_development/20261113T235717_433071Z_17991975/`,
executed from a clean clone at `7597cfbb8f6332801b5ff6baa7237967952d6233`.
All 5,132 recorded artifacts verify; the original market benchmark still
verifies all 955 artifacts. Independent audit:
`log/validation_20261114/market_audit.json`. Every completed child manifest is
clean at the same code commit. Candidate count is not fit count: monthly
replays produce multiple parameter/scaler checkpoints for one candidate.

| Asset | Lowest development RMSE policy | V1 development skill | V2 development skill | Exploratory 95% interval vs zero |
| --- | --- | --- | --- | --- |
| AAPL | frozen_expanding | +1.125% | +1.044% | [-0.489%, +2.841%] |
| MSFT | monthly_756 | +1.908% | +2.754% | [-0.841%, +5.989%] |

Both versions use the same 1,008 development targets per asset. MSFT improves
descriptively; AAPL does not. Baselines win two AAPL monthly/756 folds; they
are permitted outcomes, not suppressed trials. Selected-policy intervals both
contain zero and do not account for selecting among policies. One unselected
MSFT policy has a positive marginal interval; it is not a multiplicity-adjusted
or independent confirmation. The grid was not expanded after viewing results.
All results, including negative policies, are published in
[the measured report](../benchmarks/results/2026-11-14-adaptive-development.json)
and [figure](../benchmarks/results/2026-11-14-adaptive-development.png).

Final clean replay: `log/clean_replay/20261114T003814Z_1d109a7e/`, source
`336d9c30531fafdfc92404c4670aacfb899434bb`, independent Python 3.12 environment
and checksum-provisioned fixtures: **185 passed, 33 warnings**, no exclusions
or skipped tests; all 14 stages passed including train/verify/predict/verify.
Warnings remain upstream pandas/Keras/timezone notices. The subsequent
tie-tolerance metadata clarification passed all eight selection regressions;
it changes no numerical policy or recorded market result.

[Hosted run 43536926997](https://github.com/TranMaiQuynh/stock-forecasting/actions/runs/43536926997)
passed Linux offline lint, tests and both synthetic policy demos at `336d9c3`.
Full TensorFlow runtime verification remains the separate Windows replay;
no Linux neural-runtime coverage is claimed. Final Git/CI receipts are local
under `log/validation_20261114/` after publication.

Reassessment (0-5, evidence-based; whole repository):

| Dimension | Score | Evidence | Remaining weakness / next gate |
| --- | --- | --- | --- |
| Architecture | 3 | Existing research, walk-forward, diagnostics and artifact lifecycle reused; causal selection isolated | Legacy neural scripts still couple I/O/training/reporting |
| Code quality | 3 | Explicit policy validation, compact numerical checkpoints, targeted regressions | No complete static typing gate |
| Functional correctness | 3 | Default Ridge parity, baseline selection, refit cutoff and checkpoint reload verified | Full external/upstream workflow matrix not replayed |
| ML correctness | 3 | Chronological selection, common targets, causal refits and future-mutation tests | No independent evidence of generalization improvement |
| Reproducibility | 3 | Registered grid, clean-clone execution, immutable snapshots and source/config/artifact hashes | No cross-platform hash lock; raw snapshots/reference fixtures are local |
| Testing/reliability | 3 | 185 tests from independent checkout/venv, hosted Linux offline CI and every fit/origin audited | Linux neural runtime, GPU behavior and long-running operation unverified |
| E2E pipeline | 3 | CSV to diagnostics, candidate selection, refits, evaluation, report and saved inference | Prospective collection and delayed-label evaluation not implemented |
| Deployment | 1 | Validated offline batch CLI | No scheduler, service SLO, monitoring, rollback or execution contract |
| Documentation/DX | 3 | Learner explanations, commands, all-policy measured evidence | Exact replay needs authorized local data/artifacts |
| Practical/portfolio value | 2 | Auditable, falsifiable research and honest mixed results | No confirmed forecasting advantage, profitability or user validation |

Technical verdict: useful, reproducible research/batch software. ML Engineering
verdict: credible temporal-isolation and experiment-lifecycle work, with bounded
claims. Practical value is research/learning; market utility remains unproven.
Production readiness: **not ready**. Continue the narrow scope; do not promote
the selected policies. Next gate is a prospective collection design and data
availability, followed by actual delayed-label evidence; broader features,
neural models and execution require separate hypotheses and acceptance gates.

## Previous execution (2026-11-12)

### Continuation: reproducible research handoff

The owner requested continuation from the remaining roadmap. The starting tree
is clean on `fix/forecasting-validation` at `d0284ed`. The completed market
protocol/results below remain frozen; this continuation improves reproducibility
and offline use rather than retuning from the observed final holdout.

| Task ID | Priority | Component | Issue | Action / acceptance criteria | Verification | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R02 | P1 | Clean replay | Full runtime depends on local ignored reference artifacts | Provision exact checksum-verified fixtures into a clean local clone and independent environment; run the complete suite and offline smoke | Final replay: 142 tests, all 14 stages, end-to-end offline prediction and clean source status | DONE | `26f01f1`, `235c404`, `f89ea08` |
| R03 | P1 | Legacy scaler | Pickled 1.5.1 scaler was loaded under 1.9.1 | Export numerical scaler state under the original version; prefer validated JSON; save JSON in new training runs | 13 serializer tests; 20 runtime tests; original-version probes match at 1e-12, 3-step/12-column reference forecast max difference 0 | DONE | `58301eb`, `f99bb28`, `ab57ef2` |
| R04 | P2 | Offline consumer | Saved numerical model had no complete batch prediction command | Add verified CSV-to-next-observation prediction without fitting; enforce asset, selection cutoff and immutable source | Artifact integrity/failure tests and 7 consumer/CLI tests pass; clean-checkout prediction and verification completed | DONE | `bca5f48`, `a087dd7`, `4b300d6`, `0724efc` |
| R05 | P1 | Delivery checks | Hosted CI for this branch and clean replay were unverified | Validate committed source, publish the feature branch normally and inspect exact-SHA hosted CI | Remote SHA ddc26d0 matched local; hosted Linux offline CI succeeded; full Windows replay passed separately | DONE | `ddc26d0` |

Completion scope is an auditable offline research/batch project. Production
serving and a new market hypothesis remain separate roadmap decisions; no
positive forecasting claim or reuse of the old final holdout is authorized by
successful engineering validation.

Continuation evidence:

- `log/completion_20261112/scaler_parity.json`: numerical probes match sklearn
  1.5.1 at absolute tolerance 1e-12; reference three-step inference under 1.9.1
  with JSON has maximum absolute difference 0 over 12 numeric output columns.
  All 14 original reference files retain their hashes. Two JSON sidecars were
  added; original weights and pickle files were not rewritten.
- The first clean replay retained 45 pytest temporary-directory setup errors
  caused by access to the shared Windows temp root; it was not reported as a
  passing run. The driver now assigns per-run TMP/TEMP directories. Replay
  `log/clean_replay/20261112T050859Z_d6dc59c6/` passed all 128 then-current tests.
- Final replay `log/clean_replay/20261112T054059Z_583329fc/` checks committed
  source `0724efc24a3b3eb5355417814c897e0be241f406` in a separate clone/venv,
  with Python 3.12.16 and the constrained dependency stack: **142 passed, 24
  warnings**, no excluded/skipped tests. All 14 stages passed, including
  train/verify/predict/verify and a clean source status. Current warnings concern
  pandas/Keras deprecations and timezone plotting, not scaler version mismatch.
- The local AAPL saved-model demo is recorded in
  `log/completion_20261112/saved_forecast_demo.json`. Its origin is 2025-12-31,
  not the current market date; target timestamp/actual remain null. No model was
  refit or promoted, and the market benchmark results were not retuned.
- The old complete market run still passes strict inventory/checksum
  verification for all 955 recorded artifacts. Prior results remain descriptive
  evidence at their original code commit, rather than claims about this new
  consumer's predictive improvement.
- The feature branch was published with the owner's existing push authorization.
  Its exact remote SHA `ddc26d001f2aa9981d43c833c678a7e09b982c43` matched local;
  [hosted run 43164635037](https://github.com/TranMaiQuynh/stock-forecasting/actions/runs/43164635037)
  completed successfully. GitHub's API reported success for Linux offline
  lint/compile, tests and the synthetic demo. This is separate from the full
  Windows TensorFlow replay, not a claim of hosted neural-runtime coverage.
  All 51 post-bootstrap commits through that SHA passed the one-file, owner
  identity and commit-message audit. No generated models/datasets were tracked.
  No force push, merge to main or history rewrite was performed. Local
  publication receipts retain the final documentation SHA and CI result.

Final target-scope assessment: the offline research/batch workflow is usable
with the stated runtime and checksum-verified artifacts. Production readiness
remains **not ready**: there is no live execution, service SLO or calibrated
uncertainty claim. New neural market experiments and any new hypothesis remain
separate research tasks, with new evaluation decisions made outside the already
inspected final holdout.

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
| M01 | P1 | Market protocol | No identified market snapshot or frozen experiment specification | Freeze protocol before acquisition; retain two identified adjusted-close snapshots with hashes | 2,766 observations/asset; source/options/hash checks; final market run independently verified | DONE | `b86dcf7`, `640e908`, `9f49704` |
| W01 | P1 | Walk-forward | The new runner only had a fixed holdout | Reuse the existing runner/fold utilities; isolated child models/scalers, matched targets, final holdout after group selection | Fold isolation, no-overlap, metric recomputation, offline/failure checks; all 109 tests pass | DONE | `8f463ac`, `6032389`, `4be060f` |
| A01 | P2 | Feature ablation | Feature contribution had not been measured | Compare signed returns with signed plus absolute returns on the frozen development protocol | 8 folds/asset, 204 Ridge trials; extra group worse in development; both selected models worse on final holdout | DONE | `cfec2a4`, `9138b8c`, `9f49704` |
| H01 | P1 | Nested provenance | Parent manifests excluded all files named manifest.json, including child manifests | Exclude only the parent's own manifest; hash child manifests | Targeted nested-manifest regression and full experiment integrity checks pass | DONE | `b352fe1`, `fdc92ee` |

The scope is CPU research and short verification runs. No paid resources, live
trading or deployment are authorized. Market snapshots downloaded today may
contain retrospective vendor revisions; no point-in-time correctness is claimed.

Legacy gate evidence: `log/validation_20261112/legacy_gate.txt` records **90
passing tests**, with no exclusions, on Python 3.12.16 / TensorFlow 2.18.1 / Keras
3.15.1. The device check reports CPU only. `constraints-win-py312.txt` records
the validated 80-package version snapshot; an offline constrained-install dry
run would make no changes. It is not a cross-platform/hash-locked environment.
At that historical gate, warnings remained for a trusted local scaler fixture serialized by scikit-learn
1.5.1 and loaded under 1.9.1, and upstream pandas/Keras deprecations. Compatibility
smoke tests passed, but exact upstream result reproduction was not claimed.
R03 above subsequently resolved the scaler warning with measured parity.

The frozen market protocol was committed as `b86dcf7` before acquisition or
market scoring: two preselected assets, daily adjusted closes, deterministic
Ridge, matched targets, a predefined absolute-return feature group, chronological
development folds and a final holdout beginning 2023. It explicitly documents
survivor selection and retrospective data limitations. No result-based protocol
changes or automatic trading/model promotion are permitted in this experiment.

### Completed execution and measured outcome

The full final suite returned **109 passed, 24 warnings**, with no exclusions:
`.venv/runtime312/Scripts/python.exe -m pytest -q -o addopts= tests tests_runtime`.
Evidence: `log/validation_20261112/final_tests.txt`. Ruff and compilation checks
passed for maintained source/tests and the new benchmark runner. The extra
one-optimizer-step smoke used real v1/v6/v9 models, verified finite losses and
save/reload prediction agreement; see `log/validation_20261112/neural_smoke/`.
It is not a neural training benchmark. Checkpoint regression tests drive
controlled validation-loss callback events on real Keras models, then save/reload.

Market run: `log/market_protocol/20261112T033204_275096Z_904ed2d6/`, executed from
clean code commit `3d40d9108437f42b9115014535e89cfaf543e44d`. All 37 run manifests
were completed and clean at execution; 2,783 artifact checksum comparisons,
package source hashes, exact trial counts and independent RMSE/MAE/baseline
recalculations passed. Receipt: `log/validation_20261112/market_verification.json`.
The 204 trials are 102 per asset, not 204 neural training jobs.

Each asset has eight complete development folds and 752 final targets from
2023-01-03 to 2025-12-31. The final 124 development observations cannot form
another complete 126-target fold and are explicitly reported as an unscored
development tail; final validation can still use the pre-holdout observations.
The final model is frozen across the final holdout, with observed prior history
updating inputs at each origin. It is not retrained daily.

| Asset | Selected group | Final lookback / alpha | Final log-return RMSE | Zero-return RMSE | Final skill |
| --- | --- | --- | --- | --- | --- |
| AAPL | returns | 5 / 1 | 0.0162583524 | 0.0161022539 | -0.969420% |
| MSFT | returns | 1 / 10 | 0.0147596636 | 0.0145742149 | -1.272444% |

Development RMSE skill was positive for signed returns (AAPL 1.1250%, MSFT
1.9080%), but the extra absolute-return group performed worse in development
and was not selected. Neither selected model beat zero return on the final
holdout. Decision: no additional feature retention for this protocol and no
model promotion. Do not retune using the inspected final holdout. These results
are descriptive, not a significance test or evidence about all markets/models.

Published aggregate evidence:
[measured JSON](../benchmarks/results/2026-11-12-reference-ridge.json) and
[plot](../benchmarks/results/2026-11-12-reference-ridge.png). Raw vendor data,
snapshots, reference models and complete run logs remain local/ignored. Exact
snapshot replay is available locally; a fresh clone needs those snapshots or
new downloads, which may contain vendor revisions. Neural market benchmarking,
calibrated uncertainty, execution simulation and deployment remain unimplemented
or unverified in this new runner. Hosted CI evidence is recorded in R05 above.

### Independent reassessment after this execution

Scores concern the whole repository, including inherited components, rather
than only the verified Ridge path. Earlier assessment sections below are the
historical 2026-11-11 review and must not override this current evidence.

| Dimension | Score / 5 | Evidence | Remaining weakness / next gate |
| --- | --- | --- | --- |
| Architecture | 3 | Snapshot acquisition, offline estimator, fold orchestration and CLI have separate responsibilities | Legacy training scripts still couple I/O, training and reporting; migrate only as needed |
| Code quality | 3 | Small verified fixes, shared causal inverse, strict schema/feature-order contracts | Partial typing coverage; no repository-wide static type gate |
| Functional correctness | 3 | Runtime suite, adversarial transforms, real saved models and reload checks pass | Full upstream training/inference matrix and external competition workflows not replayed |
| ML correctness | 3 | Train-only scaling/diagnostics, nested selection, matched origins, untouched final selection tests | Retrospective data, two surviving assets, no significance/robustness inference |
| Reproducibility | 3 | Frozen protocol, pinned Windows versions, clean-clone replay, isolated environment and checksum-provisioned fixtures | No hash-locked cross-platform environment; raw datasets remain local; vendor revisions possible |
| Testing/reliability | 3 | 142 tests from a clean checkout; Linux hosted offline CI; actual optimizer/reload smoke; legacy scaler numerical parity | Full Linux neural runtime and GPU behavior unverified; third-party warnings remain |
| E2E pipeline | 3 | Acquisition to walk-forward/ablation, final scoring/report and verified saved-model CSV prediction executed | Neural training not integrated into this experiment lifecycle; only close-derived features |
| Deployment | 1 | Usable offline batch CLI with verified model lineage and explicit failure contracts | No service SLO, deployment, operational monitoring, rollback or live execution |
| Documentation/DX | 3 | Quickstart, learner guide, measured results, fixture provisioning and clean-replay command | New machine still needs authorized reference fixtures/snapshots and prepared dependency cache for exact offline replay |
| Practical/portfolio value | 2 | Defensible methodology, falsifiable contribution and honest negative results | No demonstrated final forecasting advantage, trading utility or user validation |

Technical verdict: a validated small research prototype with meaningful runtime,
evaluation and provenance improvements. ML Engineering verdict: demonstrates
temporal isolation, testing, data lineage and controlled experiments; not a
complete production lifecycle. Practical value is research/learning, while
market utility remains unproven. Production readiness: **not ready**.
Recommendation: **continue with narrow scope**, keeping the zero-return baseline.

Remaining prioritized roadmap:

1. P1, separate research scope: define one new hypothesis on development data
   only and register an
   evaluation using an actually unused final period before viewing it. Gate:
   all predefined folds/assets and negative results reported; no reuse of this
   final holdout as untouched after result-driven changes.
2. P2, conditional: if needed, integrate one existing neural model into the same lifecycle,
   with bounded CPU training and fair trial budgets. Gate: full training/selection
   and reload tests, comparable origins, complete real-run evidence.
3. P2, conditional on an actual operating requirement: measure a representative
   batch consumer's latency/resource budget before adding serving/monitoring.
   Gate: stated workload, measured resource use and operational failure policy.

R02-R05 complete the previously pending clean replay, scaler compatibility,
offline consumer and delivery gates. Later research/serving proposals are not implemented
capabilities or prerequisites for the agreed offline research/batch handoff.

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
