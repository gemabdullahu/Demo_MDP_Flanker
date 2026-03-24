function MDP = MDP_Flanker_Gen(p, template)
% POMDP generator for the Flanker full-inversion pathway.
%
% This function is the Flanker analogue of the Stroop generator used during
% full inversion. Its role is to take a clean unsolved 6-block Flanker
% template and re-parameterize it under candidate latent offsets p.c and
% p.e before passing it to the likelihood / inversion routine.
%
% IMPORTANT:
% `template` must be a clean unsolved 1x6 Flanker schedule, not an already
% solved MDP object. Earlier versions that reused solved MDP structures
% produced misleading behavior because parameter manipulations were not
% being expressed cleanly in regenerated simulations.
%
% Parameterization:
% The synthetic Flanker datasets were generated as local offsets around
% calibrated baseline values, not as absolute raw strengths. Therefore the
% generator must reconstruct effective parameter values using the same
% baseline-centered mapping:
%
%   effective c = exp(log(c0) + p.c)
%   effective e = exp(log(e0) + p.e)
%
% where:
%   p.c, p.e = latent offsets to be recovered by inversion
%   c0, e0   = calibrated baseline values used in simulation
%
% Inputs:
%   p        : struct with latent offsets
%                p.c = offset for correctness-preference strength
%                p.e = offset for policy prior / bias strength
%   template : clean unsolved 1x6 Flanker schedule
%
% Output:
%   MDP      : re-parameterized 1x6 Flanker schedule ready for inversion

% Calibrated baseline values used in synthetic-data generation.
% These define the operating point around which p.c and p.e vary.
c0 = 0.22;
e0 = 0.051366;

% Reconstruct effective strengths from latent offsets.
% This matches the baseline-centered parameterization used when the
% synthetic datasets were originally generated.
c = exp(log(c0) + p.c);
e = exp(log(e0) + p.e);

% Start from the clean schedule template.
MDP = template;

% Update each block in the 6-block Flanker schedule.
for b = 1:numel(MDP)

    % Policy prior / action bias.
    % This is the Flanker analogue of the Stroop E update.
    MDP(b).E = spm_softmax([e; -e]);

    % Preference structure associated with c.
    %
    % IMPORTANT CAVEAT:
    % This currently writes into C{4}, preserving the original 2 x T shape
    % of that modality across the whole block. Structural diagnostics showed
    % that changing c through the clean generator does affect forward
    % behavior, so this mapping is not silent. However, full inversion still
    % does not reliably recover the true generating c, so the semantic role
    % of this preference mapping in the likelihood pathway remains an open
    % issue.
    %
    % The repmat call preserves the original time-dependent shape expected
    % by the Flanker block model.
    MDP(b).C{4} = repmat([c; -c], 1, MDP(b).T);
end
end