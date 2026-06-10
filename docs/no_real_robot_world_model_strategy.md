# No-Real-Robot Submission Strategy

## Position

The paper should not claim real-robot deployment. The safer ICLR framing is:

> We study failure-aware verification of proposed or imagined robot futures. Visual plausibility and policy likelihood are insufficient; compact execution-envelope evidence identifies physically executable futures with few target-task calibrations.

This is compatible with not owning hardware because the central claim is about **embodiment-grounded verification**, not about transferring a policy to a physical robot.

## Why This Can Still Be Publishable

Real-robot experiments are valuable, but they are not the only acceptable evidence if the benchmark itself is designed around execution-grounded evaluation. The current result already has a strong mechanism:

- ordinary no-demo replay pools expose an under-actuation prior;
- energy-matched hard negatives remove the action-magnitude shortcut;
- action-only critics stay weak after that shortcut is removed;
- compact EEF/gripper execution-envelope statistics nearly close the hard-negative gap;
- no-task-ID source-only transfer stays weak at 25.6/64;
- full-source plus one target case reaches 46.0/64, and eight target cases reach 62.2/64;
- task-specific heads do not beat shared one-hot adaptation.

The result is counterintuitive but defensible: the useful signal is not more temporal detail, more object state, or a bigger task-specific head. It is few-shot contact-regime calibration over compact execution envelopes.

## Main Risk

The current strongest probe uses low-dimensional simulator traces. A reviewer can reasonably object:

> This is a privileged-state diagnostic, not a deployable world model or robot policy.

The answer should not be to overclaim. The next layer should convert the diagnostic into a world-model benchmark protocol where the input is a proposed/generated future and the output is an executable-future verification score.

## Recommended Benchmark Stack

### Layer 1: RoboCasa365 Mechanism

Use the current RoboCasa365 result as the controlled mechanism study:

- modern household manipulation benchmark;
- executable simulator;
- energy-matched hard negatives;
- rank/candidate randomization controls;
- state-key ablations;
- summary-statistic ablations;
- few-shot adaptation curves.

Claim: compact execution-envelope evidence is a high-signal verifier after action-magnitude shortcuts are controlled.

Limitation: the local downloaded RoboCasa365 target/atomic data currently contains only four demonstration tasks, so this should not be the only external benchmark layer.

### Layer 2: RoboWM-Bench Target

RoboWM-Bench is the best next benchmark if code/data are accessible. Its paper frames exactly the missing evidence: visual realism is not enough; generated manipulation futures must translate into executable actions and complete tasks. It evaluates world models through embodied action execution in simulation.

Use it to test:

- whether generated or retrieved futures that look plausible are physically executable;
- whether our execution-envelope verifier improves future selection over visual/plausibility scores;
- whether few-shot contact calibration transfers across tasks better than a task-agnostic verifier.

Ideal result shape:

| Selector | Expected role |
| --- | --- |
| visual realism / model likelihood | strong-looking but physically brittle baseline |
| action-envelope only | catches under-actuation, misses contact timing |
| no-task-ID execution-envelope verifier | weak source-only transfer |
| few-shot task/contact-calibrated verifier | improves executable success |

### Layer 3: RoboTwin 2.0 Fallback

If RoboWM-Bench code or data are not accessible quickly, use RoboTwin 2.0 as the executable benchmark fallback. It is a 2025 bimanual manipulation benchmark/data generator with many tasks, multiple embodiments, and strong domain randomization.

Use it to test:

- whether execution-envelope verification still works outside RoboCasa kitchens;
- whether the method survives stronger randomization;
- whether task/contact calibration remains sample-efficient.

### Lower Priority

RoboTrustBench is useful for video-world-model trustworthiness but is less directly tied to action reranking unless we build a generated-video scoring layer.

RoboMIND 2.0 is useful as a real-data dataset layer, but it is less immediately suited to executable reranking unless the released assets include a usable evaluation loop.

## Paper Claim Boundary

Do claim:

- proposed/generated robot futures need physical executability verification;
- execution-envelope feedback is a compact and surprisingly strong verifier;
- few target-task examples are enough to calibrate contact regimes;
- task-specific heads are not automatically better than shared task-conditioned calibration.

Do not claim yet:

- real-robot deployment;
- zero-shot cross-task world modeling;
- visual world models are solved;
- low-dimensional state traces are the final deployed input.

## Next Concrete Step

Try to obtain and smoke-test RoboWM-Bench. If unavailable, start RoboTwin 2.0 setup as the executable fallback. In parallel, convert the current RoboCasa365 result into a "future verification" API:

1. candidate future in: action trace, generated rollout, or predicted video-derived trajectory;
2. verifier score out: probability of executable task completion;
3. calibration mode: no-task-ID, task one-hot, per-task head, and few-shot target adaptation;
4. metrics: executable success, rank0 failure recovery, oracle gap, and calibration sample efficiency.
