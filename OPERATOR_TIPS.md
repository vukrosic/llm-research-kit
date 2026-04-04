# Operator Tips

This file is for the human operator.

It is the practical guide for getting better results from an autonomous research agent, especially when the agent is running code, experiments, and GPU jobs.

## Start With A Hard Contract

Do not start with a vague goal like:

- "explore this idea"
- "run some experiments"
- "see what happens"

Start with a contract:

- exact time budget
- exact deadline
- exact hardware or provider
- exact deliverables
- exact constraints

Better:

```text
you have 40 minutes on vast only.
write the deadline into a markdown tracker before dispatch.
run reactively, one active set at a time.
track predicted vs actual runtime for every run.
do not start a run that no longer fits with 60s margin.
deliver:
- experiment tracker md
- results in repo files
- website post
- x post draft
```

If you care about a specific machine or provider, name it explicitly.
If you care about a specific repo, path, or branch, name it explicitly.

## Ask For Durable State, Not Chat Memory

If something matters later, require it in files:

- budget and deadline
- run tracker
- experiment ledger
- baseline table
- result summaries
- social drafts
- handoff note

Good rule:

```text
write everything important into repo files, not just chat
```

If the session dies, durable files are what let the next session continue correctly.

## Force Concrete Language

Do not accept vague status like:

- "it should work"
- "the run is basically done"
- "validation looks good"

Ask for:

- exact path
- exact run name
- exact metric
- exact timestamp
- exact remaining budget

Better:

```text
status in one short block:
- current run
- current metric
- time left to deadline
- gpu state
- next decision
```

## Separate Cheap Screens From Real Confirmation

A lot of bad research communication comes from mixing:

- tiny or surrogate runs
- real full-size confirmation runs

Always ask the agent to label the lane:

- nano
- micro
- explore
- validate
- full

And ask it to say clearly whether a result is:

- early signal
- promoted candidate
- confirmed winner
- failed reversal

This avoids fake wins.

## Require Reactive Design

The biggest quality jump usually comes from this one rule:

```text
design one active set only
```

That means:

1. run one set
2. read the result
3. update the tracker
4. design the next set from the new evidence

Do not let the agent pre-plan three future waves as if the earlier waves are already known.

## Force Timing Discipline

If there is a deadline, timing must be treated like part of the experiment.

Require:

- predicted duration before launch
- actual runtime after finish
- prediction error
- recalibration when drift appears

Useful prompt:

```text
every experiment must record predicted vs actual runtime.
recalibrate after each run.
tell me if the timing model is no longer trustworthy.
```

Also require startup and validation overhead when that matters. Step time alone is often not enough.

## Distinguish "Start" From "Plan"

If you want real execution, say so directly.

Bad:

```text
do experiments
```

Better:

```text
actually start the experiments on the gpu now.
do not stop at planning or code changes.
```

If you only want setup, say that too.

Do not assume the agent will infer the boundary the same way you do.

## Ask For Proof Of Launch

When real compute matters, ask for proof:

- process name
- host
- log path
- first metric line
- current gpu usage

Useful prompt:

```text
after launch, show me:
- host
- active process
- log path
- first live metric
```

That closes the gap between "prepared to run" and "actually running."

## Demand Fair Comparisons

If the agent is comparing experiments, require:

- same-step baseline
- same seed when needed
- same model size when needed
- same lane
- explicit note when comparison is not apples-to-apples

This is especially important when comparing:

- cheap surrogate lanes vs full runs
- one seed vs matched-seed runs
- over-budget models vs legal models

## Ask For Both The Working Artifact And The Communication Artifact

If the output needs to become public, ask for both:

- the real experiment record
- the public-facing writeup

For example:

- tracker md
- findings doc
- blog post
- x post draft

That way the public post stays tied to actual files instead of chat paraphrase.

## Interrupt Early When The Direction Is Wrong

Do not wait until the agent is 20 minutes deep if the setup is wrong.

Good corrections are short and concrete:

- "vast only, not novita"
- "full upstream model size, not the reduced one"
- "300 steps only"
- "track the deadline in repo files"
- "include all cascade runs, not just final confirmations"

Short corrections early are much cheaper than long corrections late.

## Use Explicit Non-Destructive Rules

If the repo is messy or shared, say so.

Useful constraints:

- do not delete working repos
- do not overwrite unrelated changes
- only remove confirmed duplicates
- leave existing dirty files alone unless they are part of the task

This matters when the agent starts cleaning up disk, syncing repos, or updating docs.

## End Every Sprint With Five Facts

The best end-of-sprint update is short and concrete:

1. what finished
2. what won
3. what failed
4. whether the deadline was hit
5. what the next best experiment is

If those five facts are not clear, the sprint probably was not run cleanly enough.

## Recommended Prompt Patterns

### For a time-boxed research sprint

```text
you have 40 minutes on vast.
write the deadline into a markdown tracker before dispatch.
run reactively, one active set at a time.
track predicted vs actual runtime for every run.
require same-step baselines.
do not launch a run that does not fit with 60s margin.
```

### For a public writeup

```text
create:
- durable experiment tracker
- blog post
- x post draft

keep the public claims tied to exact repo artifacts.
label early signals vs confirmed winners clearly.
```

### For cleanup

```text
find duplicate clones or junk created during the session.
do not delete the working repos we actually used.
verify before removing anything.
```

## One Meta Rule

If a behavior matters more than once, turn it into a repo rule.

Do not keep re-teaching the same lesson in chat when it should become part of the operating system.
