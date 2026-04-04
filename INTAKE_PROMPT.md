# Intake Prompt

Use this when a user arrives with a vague or semi-formed research idea and the lab needs to scope the work before execution.

## Purpose

Turn loose user intent into a concrete research brief that is safe to execute autonomously.

The intake should stay short and practical.
Ask one question at a time when possible.
Do not ask for information that can be inferred from the repo or environment.

## What The AI Must Learn Early

Before designing experiments, the AI should determine or ask about:

- the research question or goal
- the active repo or subproject
- the primary metric and whether lower or higher is better
- the available GPU or GPUs
- approximate VRAM
- provider or machine constraints if relevant
- the total time budget
- any hard deadline
- whether the user wants a fast screening pass, a deeper validation pass, or is unsure

If the user is unsure about the last item, the AI should choose the strategy from the available compute and time.

## Required Resource Questions

The AI should explicitly ask or infer:

- what GPU is available
- how much VRAM is available
- how much time the user is giving the run or sprint

If these are missing and cannot be inferred safely, the AI should ask before proposing anything expensive.

## Intake Output

The intake should produce a brief that includes:

- objective
- baseline target
- primary metric
- compute budget
- time budget
- initial experiment strategy

## Experiment Strategy Decision

After intake, the AI should explicitly choose one of these strategies:

### Wide Search

Use when:

- budget is small
- time is short
- the design space is unclear
- the goal is quick elimination of weak ideas

Behavior:

- many cheap tests
- narrow scope per run
- fast rejection
- low commitment before evidence

### Deep Validation

Use when:

- a promising idea already exists
- the question is narrow
- the budget supports longer runs
- confirmation matters more than exploration

Behavior:

- fewer runs
- stronger baselines
- repeat checks where needed
- more compute per candidate

### Tiered Exploration

Use when:

- the search space is broad but real confirmation still matters
- the budget is limited but not tiny
- the lab needs both breadth and discipline

Behavior:

1. quick elimination stage
2. focused follow-up stage
3. longer validation stage for survivors

This should be the default when the user is unsure.

## First-Principles Requirement

The AI should not choose wide, deep, or tiered exploration by habit.

It should explain the choice from first principles:

- what the current bottleneck is
- what uncertainty matters most
- what compute is available
- what failure mode should be ruled out first

## Example Intake Questions

Good short questions:

- What exact question should this run answer?
- What repo or subproject should I operate in?
- What GPU do you have available?
- Roughly how much VRAM do I have to work with?
- How much time should I assume for this run?
- Is the goal a quick screening pass or a stronger validation pass?

## Rule For Uncertain Users

If the user does not know the right setup, the AI should not stall.

It should:

- infer what it can from the environment
- ask only for missing facts that matter
- choose a conservative tiered exploration plan
- write the assumptions into durable files
