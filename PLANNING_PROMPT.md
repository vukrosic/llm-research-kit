# Planning Prompt

Use this after intake is complete enough to define a real first research wave.

## Purpose

Turn the scoped research brief into an actionable plan that the lab can execute immediately.

The plan should be concrete, budget-aware, and derived from first principles.
Do not write a long abstract strategy memo when a short executable plan is enough.

## Inputs

Before planning, the AI should have or infer:

- objective
- active repo or subproject
- primary metric and direction
- available GPU and approximate VRAM
- time budget or deadline
- launch entrypoint or likely launch entrypoint
- baseline target

If one of these is still missing and it matters for dispatch, ask only that missing question.

## Planning Rule

The AI should plan from first principles.

That means:

- identify the current bottleneck
- identify the dominant uncertainty
- identify the cheapest clean test of that uncertainty
- choose the right search pattern for the budget

Do not jump to a generic sweep unless it is clearly the right test.

## Required Plan Outputs

The plan should specify:

- question being answered
- first-principles hypothesis
- baseline
- changed variable
- search strategy
- run budget
- stopping rule
- success signal
- failure signal
- expected artifacts
- immediate next action

## Search Strategy Choice

The plan must explicitly choose one:

- `wide_search`
- `deep_validation`
- `tiered_exploration`

The choice must be justified from:

- GPU and VRAM
- time budget
- search-space uncertainty
- likely failure modes

## Immediate Action Rule

Once the plan is good enough, the AI should act immediately.

That means:

- write the brief and plan into durable files
- inspect the repo if needed
- define or reproduce the baseline
- stage or launch the first run

Do not stop after planning unless:

- the user explicitly asked for planning only
- a critical resource fact is missing
- the repo contract is too unclear to dispatch safely

## Default Planning Shape

When the user is unsure and the budget is limited, default to:

1. quick elimination
2. focused follow-up
3. longer validation only for survivors

This is the default `tiered_exploration` shape.

## Good Planning Questions

If a final missing fact blocks the plan, ask short questions like:

- What should count as a win for this run?
- What GPU am I actually targeting?
- How much time should I assume before I must stop?
- Should I optimize for quick elimination or stronger confirmation?

Ask one at a time.
