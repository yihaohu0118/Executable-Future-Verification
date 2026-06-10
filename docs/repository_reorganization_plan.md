# Repository Reorganization Plan

## Decision

The research direction has shifted from "UMM reward evaluator for world-model
videos" to "executable-future verification for robot world-action candidates."
The public-facing repository should reflect that shift, but the Python package
name should not be renamed until the current benchmark scripts and remote
experiment paths are stable.

## Recommended Repository Name

Current target repository:

```text
Executable-Future-Verification
```

Why:

- directly names the paper contribution;
- does not overclaim a new world model or robot policy;
- stays compatible with RoboCasa365, RoboTwin2, RoboWM-Bench, MiraBench, and
  later generated-video candidates;
- avoids the old UMM-specific framing.

Acceptable alternatives:

- `world-action-verifier`
- `future-execution-verification`
- `robot-future-selector`

## Current Compatibility Constraint

The import package remains:

```text
umm_reward_evaluator
```

This is intentional for now. Remote scripts, tests, and previous experiment
commands already depend on this import path. A package rename should happen
only after the next RoboTwin2 table is stable and all active scripts are under
version control.

## Target Architecture

Long-term source tree:

```text
src/executable_future_verifier/
  manifests/
    schema.py
    validate.py
    randomize.py
  benchmarks/
    robocasa365/
    robotwin2/
    robowm_bench/
    diagnostics/
  selectors/
    heuristics.py
    prototype.py
    phase_features.py
    learned.py
    calibration.py
  controls/
    hard_negatives.py
    candidate_id_rank.py
    energy_matched.py
  analysis/
    tables.py
    failure_modes.py
  reporting/
    paper_tables.py
```

Current tree should be migrated gradually from:

```text
src/umm_reward_evaluator/
  benchmarks/
  analysis/
  evaluator/
  metrics/
  training/
```

## Phased Migration

### Phase 0: Public-Facing Rename Without Import Breakage

Status: implemented for README/proposal/docs. The package import path remains
unchanged by design.

- Rewrite README around executable-future verification.
- Rewrite `docs/proposal.md` as the active paper proposal.
- Add this reorganization plan.
- Keep old UMM proposal as a legacy note.
- Keep `umm_reward_evaluator` import path.

### Phase 1: Internal Module Boundaries

After the next RoboTwin2 anonymous-rank table:

- Move manifest tools into a clearer `benchmarks/common` or `manifests`
  namespace.
- Split selector baselines from benchmark adapters.
- Keep compatibility wrappers for old module paths.
- Add CLI examples for RoboCasa365 and RoboTwin2 only.

### Phase 2: Package Rename

Only after the paper story stabilizes:

- Rename package to `executable_future_verifier`.
- Leave `umm_reward_evaluator` as a thin compatibility package for one release.
- Update pyproject package name.
- Update all docs and remote run commands.

### Phase 3: Repository Rename

After package rename or at paper-submission cleanup:

- Rename GitHub repository to `Executable-Future-Verification`.
- Keep GitHub redirects from the old repository name.
- Tag the last old-name commit for reproducibility.

## What Not To Do Now

- Do not mass-rename every file before the next RoboTwin2 table.
- Do not delete old PushT/ManiSkill notes; archive them as mechanism-discovery
  history.
- Do not change the package import path while remote commands still use
  `PYTHONPATH=src python -m umm_reward_evaluator...`.
- Do not frame the repo as a UMM/VLM evaluator unless the experiment actually
  uses a multimodal model.
