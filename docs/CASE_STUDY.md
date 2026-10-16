# One question, two rounds, a traceable report

[中文案例](CASE_STUDY_zh.md) · [Back to the project](../README.md) · [Complete report](example-report.md) · [Exported JSON](example-report.json)

This walkthrough follows the repository's [fixed synthetic example](../examples/offline_debate.json). Its arguments, concessions, scores, stop reason, and timestamp are prewritten illustrations. It demonstrates how to inspect a report; it is not a live model run or evidence that the proposed workplace policy works.

## The question

> Should a fictional 20-person software team trial a 32-hour, four-day week at unchanged pay?

The scenario includes weekday customer support and different responsibilities across roles. The decision is whether and how to run a bounded trial. There are no observed workplace outcomes in this example.

![The current application displaying the fixed synthetic report](images/offline-debate.png)

Captured on **2026-10-16**, using application baseline `e80df34`, the bundled example, and an empty temporary history. No credentials, model execution, or detector downloads were used. The example did not add a history record.

## How the positions change

The table summarizes the six complete arguments stored in the report. Frame Challenger is the `devil` role, labelled Devil's Advocate in the interface.

| Perspective | Round 1 | Round 2 |
|---|---|---|
| Optimist | Start a reversible eight-week pilot and review delivery, coverage, and workload. | Assign support owners and review thresholds before starting; inspect workload by role. |
| Pessimist | Retain the current schedule until the coverage and measurement plans are ready. | Accept a reversible pilot after a representative baseline and a support-rota rehearsal. |
| Frame Challenger | Decide how time off and coverage duties can be allocated fairly. | Require role-level agreement and check who absorbs handovers and less desirable days off. |

In the app, open **Full Reasoning Trace** to inspect each position, its reasoning, key claims, self-reported confidence, and any concession. Both rounds and all three roles are also preserved in the [complete exported report](example-report.md#reasoning-trace).

## Two concessions with a source

The report stores both the changed position and the specific earlier claim that triggered it.

1. **The Optimist adds a coverage prerequisite.** The Pessimist's first-round claim was: “Weekday customer coverage must be assigned before working hours change.” The Optimist then concedes that the pilot should wait for named support owners, because informal overtime could otherwise replace the apparent schedule savings.
2. **The Pessimist accepts a bounded experiment.** The Optimist's first-round claim was: “An eight-week pilot can be reversed if pre-agreed delivery or coverage thresholds are missed.” The Pessimist then concedes that a reversible pilot is preferable to rejecting every experiment, because explicit stop criteria limit the risk of an unsuccessful policy.

![The application showing both concessions, their triggering claims, and rationales](images/concessions.png)

The **Concession Log** displays these claims and rationales. The same records appear under the relevant second-round arguments and in the exported global log. A reader can check whether a recorded concession actually addresses the cited claim.

## What the report concludes—and leaves open

The final recommendation is to prepare a reversible eight-week pilot with unchanged pay, a baseline, named weekday support owners, and stop criteria. Delivery, customer coverage, and the distribution of workload should be reviewed before considering expansion.

The discussion still leaves two disputes visible:

- **Readiness:** assigning owners may be enough for the Optimist, while the Pessimist also requires a representative baseline and rota rehearsal; the Challenger requires agreement by role.
- **Wider adoption:** a short pilot could motivate another trial, but may miss seasonal demand or hide unequal burdens across roles.

The saved stop reason is `max_rounds`, displayed as **Round limit reached**. It does not claim that every disagreement was resolved or that the recommendation was verified.

## Why the index is 18.0 / 100

The example uses illustrative divergence values of `0.8` and `0.2`. The report's documented formula uses the highest value across both rounds:

```text
100 × (1 − max(0.8, 0.2)) × 0.9 = 18.0 / 100
```

This is the **Debate Agreement Index**, not an 18% probability of a correct answer. The detector did not produce these example values. Agent confidence values are also illustrative and do not enter this formula. See the [full interpretation](../README.md#reading-the-debate-agreement-index) for detector and historical-report limitations.

## Open the same case

After the [installation steps](../README.md#quick-start), run `streamlit run app.py` and click **View offline example**. No API key is required. Expand **Full Reasoning Trace**, **Concession Log**, and the disputed-point panels, then use **Download Markdown** or **Download JSON** to keep the report.

Browsing and exporting the example do not execute the debate graph, download models, or save it into **Past Debates**. Starting a new debate is a separate action that requires configured credentials.

The checked-in [Markdown](example-report.md) and [JSON](example-report.json) were generated from `examples/offline_debate.json` using the application's `build_report_exports(..., source="sample", provenance=...)`. They retain the original report values and synthetic-source label. The JSON uses the same versioned envelope as the app download.
