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
