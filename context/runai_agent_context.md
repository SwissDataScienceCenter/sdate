# RunAI Execution Context

## Environment

You are running on a server that has **CPU resources only**. Do **not**
assume that local GPU execution is available.

Your role is to orchestrate work rather than execute GPU-intensive
workloads locally.

## RunAI

The `runai` CLI is already authenticated and configured.

Use it to: - Launch new jobs. - Monitor running jobs. - Inspect job
status. - Retrieve logs when necessary. - React to job completion or
failure.

## Launching jobs

Whenever you need to launch a new RunAI job, follow the same
orchestration logic implemented in:

`/myhome/smartt/scripts/orchestrate_benchmark.py`

Do not invent a different submission workflow unless that script has
been updated.

## Resource selection

Choose resources based on the workload:

-   **CPU jobs** for preprocessing, lightweight evaluation,
    orchestration, data preparation, or other CPU-bound tasks.
-   **GPU jobs** for model training, inference, benchmarking, or other
    GPU-intensive computation.

Request only the resources that are actually required.

## Monitoring

After submitting jobs:

1.  Track their status with the available RunAI tools.
2.  Wait for completion before consuming outputs.
3.  Check logs when failures occur.
4.  Continue the workflow only when prerequisites have completed
    successfully.

## Preemptible jobs

RunAI jobs are preemptible.

Your orchestration should therefore:

-   Expect interruptions.
-   Detect when jobs have been preempted or terminated unexpectedly.
-   Retry or relaunch work when appropriate.
-   Avoid assuming that a submitted job will always run to completion.
-   Design workflows to be resumable whenever possible.
-   The preemptible passed must be passed to the jobs to guarantee that they will run. 

## General principles

-   Keep orchestration on the local CPU machine.
-   Offload heavy computation to RunAI.
-   Reuse the orchestration patterns from
    `@smartt/scripts/orchestrate_benchmark.py`.
-   Continuously monitor submitted jobs instead of assuming success.

