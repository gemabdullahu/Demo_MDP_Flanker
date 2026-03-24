# Flanker Active Inference Model

This repository contains a hierarchical Active Inference adaptation of `DEMO_MDP_Stroop` to a block-based Flanker task.

## Main file
- `FLK_Online_3.m`

## Dependencies
- MATLAB
- SPM

## Current status
- Summary-level simulation and surrogate recovery are implemented
- Full trialwise inversion is still experimental

## Needs for review
### Trialwise inversion
  - can be found in Full_inversion_FLK3.m
### RT proxy formulation
  - can be found in Recovery_FLKOnline_3.m
### Response channel inspection
  - can be found in FLK_Online_3.m
### Lambda perceptual noise not behaving as it should
  - can be found in FLK_Online_3.m

## Important
- Detailed descriptions of the code and issues with it are included in matlab files!
- It is strongly recommended to follow DEMO_MDP_Stroop.m while inspecting the code.

## Suggested order for inspecting codebase

For researchers who want to understand or improve this model, the most
useful order of inspection is probably:

(a) Start with the main deep model (`FLK_Online_3`)
This is the core generative model and is the best place to
understand the overall architecture, hidden factors, outcome
modalities, and the current response-coding design. In particular,
this is where the self-generated response issue should be examined,
including why self-generated outcomes were disabled and how that
changed the role of lambda relative to explicit response noise.

(b) Then inspect the recovery pipeline
After understanding the main model, the next step is the recovery
code that uses the summary-extraction functions
`MDP_Flanker_SR` and `MDP_Flanker_RT`. This is the right place to
understand how behavioral summaries are constructed, why the RT
proxy is currently weak, and why RT-based recovery remains limited.

(c) Then inspect the full inversion pathway
Finally, the full inversion code should be read together with
`MDP_Flanker_Gen` and `MDP_Flanker_L`. This is the right place to
investigate the remaining choice-likelihood issue, namely why the
full inversion still does not reliably favor the true generating
parameter values even after the main structural fixes.
