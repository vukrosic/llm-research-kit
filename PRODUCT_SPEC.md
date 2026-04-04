# Product Spec

This file defines the first shippable product for `Open Research Loop`.

The repo already defines the lab operating system.
This file defines the first concrete user-facing workflow the system must perform well.

## Product Statement

The first product is:

- a user brings a research question
- the AI scopes it through short questions
- the AI writes a concrete research plan into repo files
- the user can pull away
- the AI executes the research loop autonomously
- the AI writes back results, next steps, and open questions

This is the first unit of value.

## Why This Is The First Product

The lab vision is large.
The first product must be much smaller.

It must prove five things:

1. vague intent can be turned into a scoped research brief
2. the brief can be turned into an executable plan
3. the AI can run real work, not just suggest ideas
4. the AI can update the plan from evidence
5. the work leaves behind durable files another person or agent can continue from

## What The First Product Is Made Of

Version 1 is mostly:

- prompts
- file conventions
- a repo integration contract
- a small set of operating rules

It is not yet:

- a hosted platform
- a multi-tenant scheduler
- a paper-writing machine
- a general autonomous science system

## Research Style

The AI should design experiments from first principles.

That means:

- start from the objective, metric, data, compute limits, and code constraints
- identify the mechanism that might explain success or failure
- derive hypotheses from that mechanism
- design the smallest experiment that tests the hypothesis cleanly

Use prior literature, common ablations, and standard recipes as references, not as the default agenda.
If a standard experiment is reused, the AI should be able to explain why it is the correct test in this repo and under this budget.

## Preferred Entry Point

The preferred first entry point is:

- an existing repo with runnable code

This is better than starting from a paper with no code because:

- the AI can inspect real constraints
- the baseline can be grounded in actual code
- experiments can start sooner
- the result is easier to verify

Later, the system can support paper-first entry points.
That is not the first product.

## First User Flow

### Stage A: Intake

The user gives a research direction or question.

Examples:

- best activation function for this LLM
- reproduce a claim in this repo
- test a small architecture change

The AI should ask short questions, one at a time, until it can safely define:

- objective
- active codebase
- available compute
- available GPU and approximate VRAM
- acceptable run budget
- time budget and any hard deadline
- baseline target
- first experiment lane

If the user does not know a design choice, the AI should mark it as a research variable, not a blocker.
If GPU, VRAM, or time budget are unknown and cannot be inferred safely, the AI should ask before proposing expensive work.

### Stage B: Brief

The AI writes the scoped brief into durable files.

Minimum durable outputs:

- `goals/ACTIVE.md`
- `goals/<goal>/MISSION.md`
- `goals/<goal>/goal.json`
- `goals/<goal>/plans/year.md`
- `goals/<goal>/plans/<current-week>.md`
- `state/NOW.md`

The user should be able to stop talking at this point and still leave the system in a runnable state.
Once the brief is good enough, the AI should move directly into planning and execution.

### Stage C: Plan

The AI converts the brief into an actionable first research wave.

The plan must specify:

- first-principles hypothesis
- baseline
- changed variable
- run budget
- strategy choice: wide search, deep validation, or tiered exploration
- stopping rule
- expected artifact paths
- what counts as a useful result even if the experiment fails

The strategy choice should be justified from:

- available GPU and VRAM
- time budget
- search-space uncertainty
- expected failure modes

Once the plan is good enough, the AI should not wait for extra prompting.
It should proceed to the first concrete action unless a real blocker exists.

### Stage D: Execution

The AI inspects the repo, runs the baseline if needed, launches the first experiment, and logs what happened.

The AI should operate autonomously unless it hits:

- missing credentials or missing data
- unclear launch or parse contract
- major repo ambiguity
- a strategic fork that changes the mission

### Stage E: Result

The AI writes back:

- what was tried
- what happened
- what was learned
- whether the result changed the plan
- the next best experiment

Minimum durable outputs:

- experiment metadata under `experiments/`
- a report under `reports/`
- updated knowledge files
- updated `state/NOW.md`

## Step-Based Rollout

These steps are maturity gates for the product, not the general lab vision.

### Step 1: One Successful End-To-End Run

Goal:
- prove the loop works once

Definition of done:
- one user question becomes a scoped brief
- the AI writes the plan to repo files
- the AI executes at least one real experiment or controlled run
- the AI writes a result summary
- the user did not need to drive each step manually
- the AI did not stall between intake, planning, and execution

What this proves:
- the system is more than a prompt demo

### Step 2: Repeatable Narrow Research

Goal:
- prove the loop works repeatedly, not just once

Definition of done:
- at least 3 separate narrow research runs complete end-to-end
- they use the same file protocol
- the AI handles at least one failed run cleanly
- the AI updates the next plan from evidence

What this proves:
- the system is a usable workflow, not a lucky run

### Step 3: External Pilot

Goal:
- prove other serious users can use it

Definition of done:
- 2 to 5 outside researchers complete their own runs
- they need limited direct support
- their usage reveals concrete workflow gaps
- the kit improves from their feedback

What this proves:
- the system is no longer only the founder's personal process

### Step 4: One Credible Public Artifact

Goal:
- prove the workflow can generate something worth sharing

Definition of done:
- one narrow result is clean enough to publish publicly
- the artifact includes plan, method, results, and limitations

Good targets:
- technical memo
- arXiv note
- workshop submission

## What The AI Must Do Well In V1

The AI must be good at:

- asking short clarifying questions
- identifying missing constraints
- deriving hypotheses from first principles
- turning chat into durable state
- inspecting repo structure
- finding the training and evaluation entrypoints
- choosing between wide search, deep validation, and tiered exploration
- proposing controlled experiments
- running actual commands
- writing results concisely

The AI does not need to be good at:

- managing many simultaneous users
- operating a large cluster
- replacing a research lead

## Prompt Stack

The first product is prompt-heavy on purpose.

It should have at least five prompt roles:

1. intake prompt
2. planning prompt
3. autonomous execution prompt
4. result analysis prompt
5. recovery prompt

The current repo already has part of this:

- `INTAKE_PROMPT.md`
- `PROMPTS_AUTONOMOUS.md`
- `PROMPTS_SUPERVISED.md`

## Repo Contract For V1

A target repo is usable when the AI can determine:

- how to start a run
- how to detect completion
- how to find logs
- how to identify the primary metric
- where result artifacts are written

If those cannot be inferred safely, the AI must ask.

## Relationship To The Rest Of This Repo

This file sits below the high-level vision and above the operational templates.

- `README.md` explains the system and why it exists
- `LAB.md` defines authority and rules
- `OPERATING_MODEL.md` defines the research loop
- `PRODUCT_SPEC.md` defines the first concrete product behavior
- `FOLDER_BLUEPRINT.md` and `TEMPLATES.md` define the durable file protocol
- `PROMPTS_*.md` provide starter prompt entrypoints

## Current Product Decision

For now, the product should optimize for:

- existing repo entrypoints first
- narrow research questions first
- autonomous execution after brief scoping
- durable written outputs after every run

This should remain the focus until Step 2 is complete.
