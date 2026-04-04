# Operating Model

This document describes how the lab runs once it is installed into a real repo.

Hard policy lives in `LAB.md`.
This file focuses on execution mechanics.

## Core Objects

The lab revolves around five durable objects:

1. goals
2. project configs
3. experiment records
4. knowledge files
5. handoff and reporting files

All of them should live inside the target repo.

## Planning Hierarchy

Use this cascade:

1. `MISSION.md`
2. `plans/year.md`
3. `plans/q<N>_<YYYY>.md`
4. `plans/<YYYY>_<MM>.md`
5. `plans/<YYYY>_w<WW>.md`
6. campaign and wave docs
7. experiment briefs and snapshots

High-level intent cascades downward.
Findings cascade upward.

## Standard Research Loop

1. Reconcile state from durable files.
2. Read the active goal and current plans.
3. Review knowledge so failed ideas are not repeated.
4. Derive the current bottleneck and candidate hypotheses from first principles.
5. Check running experiments.
6. Collect completed results.
7. Compare against the correct same-step baseline.
8. Reject, revalidate, or promote.
9. Update knowledge and reports.
10. Design the next reactive set from first-principles hypotheses, not only standard recipes.
11. Dispatch or stage the next work.
12. Write `state/NOW.md` before ending the session.

## Experiment Record Requirements

Every experiment should record:

- name
- hypothesis
- first_principles_rationale
- project
- parent base
- stage
- step count
- baseline metric
- promotion threshold
- expected duration
- predicted duration
- prediction source
- prediction sample count
- predicted startup / validation / post-train overhead when available
- change summary

Every completed experiment should also record:

- primary metric
- runtime
- prediction error
- prediction error ratio
- actual startup / validation / post-train overhead when available
- steps completed
- environment used
- relevant log tail or artifact pointer

## Timing Model

Before launching a run, the lab should have a machine-readable estimate for:

- remaining budget seconds
- predicted duration for the next run
- projected remaining sweep time after that run

After each completed run, the lab should update:

- actual runtime
- prediction error
- remaining budget
- whether recalibration is needed

Batch plans should be revised after each completed run, not only at the start of a sweep.

## Reactive Design

The lab should design only one active set at a time.

That means:

- run the current set
- read the results
- update timing and knowledge
- only then design the next set

Do not queue speculative future sets unless the human explicitly asks for that behavior.

## Baseline Discipline

Never evaluate an experiment against a mismatched baseline.

If the lab wants to test at a new step count, it must first produce a baseline for that same step count using the unmodified base.

## Repo Integration Contract

A target repo is usable when the lab can determine:

- how to launch a run
- how to detect whether a run is still active
- where logs go
- where result artifacts appear
- how the primary metric is parsed

These are project-level details, not template-level details.

## Source-Of-Truth Priority

Recommended priority:

1. experiment records
2. project config
3. goal and plan files
4. knowledge files
5. state dashboards

Dashboards can be regenerated.
Primary records should not be inferred after the fact if avoidable.

## Session End Rule

Every session should leave a clean handoff:

- what is running
- what finished
- what was learned
- what should happen next
- what is blocked
