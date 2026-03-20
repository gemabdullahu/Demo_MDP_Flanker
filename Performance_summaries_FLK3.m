%% =========================================================
% FLK_SANITYCHECK_CONGRUENCY_BY_CONTEXT.m
% =========================================================
% PURPOSE
% -------
% This script provides a behavioral sanity check for the hierarchical
% Flanker active inference model implemented in `FLK_Online_3`.
%
% It is the Flanker analogue of the Stroop "Congruency x task condition"
% sanity-check step: before fitting or recovery, it verifies that the model
% produces sensible context-dependent behavioral patterns across:
%
%   1. MC = mostly congruent context
%   2. MI = mostly incongruent context
%   3. NP = non-predictive / balanced context
%
% The script simulates one block from each context using a fixed/default
% parameter setting and extracts:
%
%   - accuracy on congruent trials
%   - accuracy on incongruent trials
%   - RT-like proxy values on congruent trials
%   - RT-like proxy values on incongruent trials
%
% It then summarizes the 6 condition means:
%
%   1. MC congruent
%   2. MC incongruent
%   3. MI congruent
%   4. MI incongruent
%   5. NP congruent
%   6. NP incongruent
%
%
% CURRENT OUTPUT UNDER DEFAULT PARAMETERS
% ---------------------------------------
% With the current default / calibrated parameter setting, the script
% produces the following condition-wise summaries:
%
%   MC congruent   acc = 0.953 | RT = 0.689
%   MC incongruent acc = 0.875 | RT = 0.982
%
%   MI congruent   acc = 1.000 | RT = 0.666
%   MI incongruent acc = 0.936 | RT = 0.993
%
%   NP congruent   acc = 1.000 | RT = 0.695
%   NP incongruent acc = 0.938 | RT = 1.005
%
% These outputs show the expected qualitative pattern:
%
%   - congruent trials are more accurate than incongruent trials
%   - incongruent trials have slower RT-like values than congruent trials
%   - the model reproduces the broad direction of the empirical task
%     behavior observed in the real data
%
%
% INTERPRETATION OF THE CURRENT PATTERN
% -------------------------------------
% The current sanity-check output is broadly consistent with the empirical
% Flanker task:
%
%   - MC shows the clearest congruency cost in accuracy
%   - all contexts show slower RT-like values for incongruent trials
%   - congruent trials remain near ceiling, especially in MI and NP
%
% In the current implementation, MI and NP congruent conditions are
% particularly difficult to move away from ceiling. A likely practical
% reason is that the number of congruent trials in some contexts is
% relatively limited and the current parameterization does not strongly
% perturb those already-easy trials.
%
% As a result:
%
%   - MI congruent and NP congruent can remain at or near 100% accuracy
%   - most informative variation is carried by the incongruent conditions
%
% This is one reason later recovery analyses often find that
% `MC_incong`, `MI_incong`, and `NP_incong` carry most of the useful
% parameter sensitivity.
%
%
% ETA SENSITIVITY
% ---------------
% The motor-noise parameter `eta` plays an important role in shaping the
% output.
%
% In the present model:
%
%   - small `eta` preserves the expected qualitative trends
%   - increasing `eta` can soften the response mapping
%   - however, if `eta` is increased too much, it disturbs the behavioral
%     pattern more strongly and can wash out or compress the intended
%     congruency/context effects
%
% Therefore, the default value used here (`eta = 0.10`) should be understood
% as a practical compromise: large enough to avoid an unrealistically rigid
% response channel, but not so large that it heavily distorts the expected
% task trends.
%
%
% MAIN USE
% --------
% This script is intended as a quick model validation / sanity-check tool.
% It should be run before summary-based surrogate recovery or fitting, to
% confirm that the current parameterization of the Flanker model produces
% qualitatively sensible behavior.
%
%
% MODEL DEPENDENCIES
% ------------------
% This script requires the following functions to be available on the MATLAB
% path:
%
%   - FLK_Online_3.m
%   - MDP_Flanker_RT.m
%
% and it also assumes access to SPM active inference functions, especially:
%
%   - spm_MDP_VB_X
%
%
% HOW THE SCRIPT WORKS
% --------------------
% 1. A default parameter structure `p` is defined.
% 2. A template deep MDP is built with `FLK_Online_3(p)`.
% 3. Three single-context simulations are run:
%       - MC block
%       - MI block
%       - NP block
% 4. Observed trial outcomes are extracted from each solved MDP.
% 5. For each context, trials are split into:
%       - congruent
%       - incongruent
% 6. Accuracy and RT proxy summaries are computed for each of the 6 cells.
% 7. The script plots:
%       - percentage correct by condition
%       - RT histograms by condition
%       - mean RT by condition
% 8. Numeric summaries are printed to the MATLAB command window.
%
%
% PARAMETER DEFINITIONS
% ---------------------
% The current default / calibrated parameter values used here are:
%
%   p.pc  = [0.75 0.25 0.50]
%       Context-specific congruency probabilities:
%       - MC: 75% congruent
%       - MI: 25% congruent
%       - NP: 50% congruent
%
%   p.c   = log(0.22)
%       Strength of preference for correct over incorrect outcomes.
%
%   p.e   = log(0.051366)
%       Prior bias over policies / response tendencies.
%
%   p.eta = 0.10
%       Explicit motor noise parameter controlling the stochasticity of the
%       final response mapping.
%
%   p.run = false
%       Returns a template model from `FLK_Online_3`, which is then solved
%       explicitly in this script via `spm_MDP_VB_X`.

clearvars -except OPTIONS
clc;

if ~exist('OPTIONS','var')
    OPTIONS = struct('gamma',1,'plot',0);
end

%% ---------------------------------------------------------
% Current default / calibrated parameters
% ---------------------------------------------------------
p = struct();
p.run = false;
p.pc  = [0.75 0.25 0.50];
p.c   = log(0.22);
p.e   = log(0.051366);
p.tau = 8;
p.eta = 0.10;  

mdp0 = FLK_Online_3(p);

% MC
mdp_MC = mdp0;
mdp_MC.T = 60;
mdp_MC.D{2} = [1;0;0];
rng default
MDP_MC = spm_MDP_VB_X(mdp_MC, OPTIONS);

% MI
mdp_MI = mdp0;
mdp_MI.T = 60;
mdp_MI.D{2} = [0;1;0];
rng default
MDP_MI = spm_MDP_VB_X(mdp_MI, OPTIONS);

% NP
mdp_NP = mdp0;
mdp_NP.T = 30;
mdp_NP.D{2} = [0;0;1];
rng default
MDP_NP = spm_MDP_VB_X(mdp_NP, OPTIONS);

%% ---------------------------------------------------------
% Determine behavioural measures from outcomes
% H rows:
%   1 = congruent?
%   2 = correct?
%   3 = RT proxy
% ---------------------------------------------------------
trMC = MDP_MC.mdp(2:end);
trMI = MDP_MI.mdp(2:end);
trNP = MDP_NP.mdp(2:end);

% ---------- MC ----------
oF = arrayfun(@(x) x.o(1,2), trMC);
oT = arrayfun(@(x) x.o(2,2), trMC);
oR = arrayfun(@(x) x.o(4,2), trMC);

valid = (oF~=3) & (oT~=3) & (oR~=3);
H_MC = nan(3,sum(valid));

oFv = oF(valid); oTv = oT(valid); oRv = oR(valid);
H_MC(1,:) = (oFv == oTv);        % congruent?
H_MC(2,:) = (oRv == oTv);        % correct?
rtMC = MDP_Flanker_RT(MDP_MC);
H_MC(3,:) = rtMC(1:sum(valid));  % RT proxy

% ---------- MI ----------
oF = arrayfun(@(x) x.o(1,2), trMI);
oT = arrayfun(@(x) x.o(2,2), trMI);
oR = arrayfun(@(x) x.o(4,2), trMI);

valid = (oF~=3) & (oT~=3) & (oR~=3);
H_MI = nan(3,sum(valid));

oFv = oF(valid); oTv = oT(valid); oRv = oR(valid);
H_MI(1,:) = (oFv == oTv);
H_MI(2,:) = (oRv == oTv);
rtMI = MDP_Flanker_RT(MDP_MI);
H_MI(3,:) = rtMI(1:sum(valid));

% ---------- NP ----------
oF = arrayfun(@(x) x.o(1,2), trNP);
oT = arrayfun(@(x) x.o(2,2), trNP);
oR = arrayfun(@(x) x.o(4,2), trNP);

valid = (oF~=3) & (oT~=3) & (oR~=3);
H_NP = nan(3,sum(valid));

oFv = oF(valid); oTv = oT(valid); oRv = oR(valid);
H_NP(1,:) = (oFv == oTv);
H_NP(2,:) = (oRv == oTv);
rtNP = MDP_Flanker_RT(MDP_NP);
H_NP(3,:) = rtNP(1:sum(valid));

%% ---------------------------------------------------------
% Condition summaries 
% Order:
%   1 MC congruent
%   2 MC incongruent
%   3 MI congruent
%   4 MI incongruent
%   5 NP congruent
%   6 NP incongruent
% ---------------------------------------------------------
MC_cong   = mean(H_MC(2,H_MC(1,:)==1), 'omitnan');
MC_incong = mean(H_MC(2,H_MC(1,:)==0), 'omitnan');

MI_cong   = mean(H_MI(2,H_MI(1,:)==1), 'omitnan');
MI_incong = mean(H_MI(2,H_MI(1,:)==0), 'omitnan');

NP_cong   = mean(H_NP(2,H_NP(1,:)==1), 'omitnan');
NP_incong = mean(H_NP(2,H_NP(1,:)==0), 'omitnan');

B = [ ...
    MC_cong; ...
    MC_incong; ...
    MI_cong; ...
    MI_incong; ...
    NP_cong; ...
    NP_incong ...
    ];

Rbar = [ ...
    mean(exp(H_MC(3,H_MC(1,:)==1) + randn(size(H_MC(3,H_MC(1,:)==1)))/16)/2, 'omitnan'); ...
    mean(exp(H_MC(3,H_MC(1,:)==0) + randn(size(H_MC(3,H_MC(1,:)==0)))/16)/2, 'omitnan'); ...
    mean(exp(H_MI(3,H_MI(1,:)==1) + randn(size(H_MI(3,H_MI(1,:)==1)))/16)/2, 'omitnan'); ...
    mean(exp(H_MI(3,H_MI(1,:)==0) + randn(size(H_MI(3,H_MI(1,:)==0)))/16)/2, 'omitnan'); ...
    mean(exp(H_NP(3,H_NP(1,:)==1) + randn(size(H_NP(3,H_NP(1,:)==1)))/16)/2, 'omitnan'); ...
    mean(exp(H_NP(3,H_NP(1,:)==0) + randn(size(H_NP(3,H_NP(1,:)==0)))/16)/2, 'omitnan') ...
    ];

labels = {'MC cong','MC incong','MI cong','MI incong','NP cong','NP incong'};

%% ---------------------------------------------------------
% Plot performance summaries in each condition
% ---------------------------------------------------------
figure('Color','w','Position',[100 80 1000 1000]);

subplot(3,1,1)
Xcat = categorical(labels);
Xcat = reordercats(Xcat, labels);

bar(Xcat, B*100, 'FaceColor',[.7 .7 .9], 'EdgeColor',[1 1 1])
axis square
ylabel('% correct')
title('Percentage correct')

%% ---------------------------------------------------------
% Plot reaction time distributions in each condition
% ---------------------------------------------------------
subplot(3,1,2); cla; hold on

bins = 0.4:0.02:1.2;

histogram(exp(H_MC(3,H_MC(1,:)==1) + randn(size(H_MC(3,H_MC(1,:)==1)))/16)/2, bins, 'FaceAlpha', 0.35)
histogram(exp(H_MC(3,H_MC(1,:)==0) + randn(size(H_MC(3,H_MC(1,:)==0)))/16)/2, bins, 'FaceAlpha', 0.35)

histogram(exp(H_MI(3,H_MI(1,:)==1) + randn(size(H_MI(3,H_MI(1,:)==1)))/16)/2, bins, 'FaceAlpha', 0.35)
histogram(exp(H_MI(3,H_MI(1,:)==0) + randn(size(H_MI(3,H_MI(1,:)==0)))/16)/2, bins, 'FaceAlpha', 0.35)

histogram(exp(H_NP(3,H_NP(1,:)==1) + randn(size(H_NP(3,H_NP(1,:)==1)))/16)/2, bins, 'FaceAlpha', 0.35)
histogram(exp(H_NP(3,H_NP(1,:)==0) + randn(size(H_NP(3,H_NP(1,:)==0)))/16)/2, bins, 'FaceAlpha', 0.35)

xlabel('Reaction Time (s)')
ylabel('Counts')
title('Reaction time distribution')
legend('MC cong','MC incong', ...
       'MI cong','MI incong', ...
       'NP cong','NP incong', ...
       'Location','northeastoutside')

%% ---------------------------------------------------------
% Third panel: mean RT by condition
% ---------------------------------------------------------
subplot(3,1,3)
bar(Xcat, Rbar, 'FaceColor',[.7 .7 .9], 'EdgeColor',[1 1 1])
axis square
ylabel('Mean RT (s)')
title('Mean reaction time')

%% ---------------------------------------------------------
% Print numeric summaries
% ---------------------------------------------------------
fprintf('\n===== FLANKER SANITY CHECK =====\n');
fprintf('MC congruent   acc = %.3f | RT = %.3f\n', B(1), Rbar(1));
fprintf('MC incongruent acc = %.3f | RT = %.3f\n', B(2), Rbar(2));

fprintf('MI congruent   acc = %.3f | RT = %.3f\n', B(3), Rbar(3));
fprintf('MI incongruent acc = %.3f | RT = %.3f\n', B(4), Rbar(4));

fprintf('NP congruent   acc = %.3f | RT = %.3f\n', B(5), Rbar(5));
fprintf('NP incongruent acc = %.3f | RT = %.3f\n', B(6), Rbar(6));


shg
drawnow
