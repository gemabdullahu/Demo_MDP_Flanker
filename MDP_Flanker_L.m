function L = MDP_Flanker_L(P,M,~,Y)
% Likelihood function for exploratory Flanker full inversion.
%
% This function is the Flanker analogue of the Stroop full-inversion
% likelihood. Given candidate latent offsets P.c and P.e, it:
%
%   1. regenerates a clean 6-block Flanker schedule under those parameters
%   2. injects the observed blockwise outcome matrices
%   3. re-solves each block with forced observations
%   4. evaluates the log-likelihood of the observed choices
%   5. optionally evaluates the RT likelihood (currently usually disabled)
%
% IMPORTANT:
% This likelihood is still considered exploratory rather than fully
% validated. Structural diagnostics showed that several earlier template and
% observation-assignment issues had to be fixed, and the present full
% inversion still does not reliably favor the true generating parameter
% values. Therefore this function should currently be interpreted as a
% diagnostic / development likelihood rather than a finalized recovery tool.

% Initialize total log-likelihood
L = 0;

% ---------------------------------------------------------
% Regenerate the full clean Flanker schedule under candidate parameters
% ---------------------------------------------------------
% M.G(P) returns a clean 1x6 block schedule re-parameterized by P.c and P.e
% using the baseline-centered generator.
mdp = M.G(P);

% ---------------------------------------------------------
% Assign observed outcomes block by block
% ---------------------------------------------------------
% Y.o is passed in as one observed outcome matrix per block:
%   Y.o{b} = observed 6 x T outcome matrix for block b
%
% These observations are injected into the regenerated schedule before
% re-solving the model.
for b = 1:numel(mdp)
    mdp(b).o = Y.o{b};
end

% ---------------------------------------------------------
% Re-solve the model with forced observations
% ---------------------------------------------------------
% Each block is solved separately under the observed outcome sequence.
% A cell array is used because different solved blocks may not always have
% perfectly identical field structures.
OPTIONS.gamma = 1;

MDPfit = cell(1, numel(mdp));
for b = 1:numel(mdp)
    MDPfit{b} = spm_MDP_VB_X(mdp(b), OPTIONS);
end

% ---------------------------------------------------------
% Choice likelihood
% ---------------------------------------------------------
% If enabled, the choice likelihood is computed from the probability the
% solved model assigns to the actually observed response on each trial.
%
% At each deep trial i:
%   - posterior state beliefs xn are extracted
%   - the predicted response distribution is obtained through A{4}
%   - the log-probability of the observed keypress is added to L
%
% IMPORTANT CAVEAT:
% Diagnostic checks indicate that this choice-likelihood term is still not
% cleanly aligned with the forward parameter-to-behavior mapping for c in
% the current Flanker full inversion. It is therefore one of the main
% remaining targets for further debugging and refinement.
if M.ch == 1
    for b = 1:numel(MDPfit)
        Mb = MDPfit{b};

        % Deep trials begin at index 2 because trial 1 is the initial setup
        for i = 2:length(Mb.mdp)

            % Extract posterior beliefs over hidden-state factors at the
            % final within-trial update step
            xn = cell(1, numel(Mb.mdp(i).xn));
            for j = 1:numel(Mb.mdp(i).xn)
                xn{j} = Mb.mdp(i).xn{j}(end,:,2,1);
            end

            % Predict response probabilities from the response modality
            % (currently assumed to be A{4})
            qresp = spm_dot(Mb.mdp(i).A{4}, xn);

            % Observed response index at the effective decision time point
            obs_r = Mb.mdp(i).o(4,2);

            % Add log-probability of the observed response
            L = L + spm_log(qresp(obs_r));
        end
    end
end

% ---------------------------------------------------------
% RT likelihood
% ---------------------------------------------------------
% If enabled, the model-predicted RT proxy is reconstructed blockwise and
% compared with the observed stacked RT vector Y.r using a Gaussian
% log-likelihood on log-RT.
%
% RT is usually disabled in the current Flanker full inversion because:
%   - the Flanker RT proxy is weak and nearly flat over the parameter grid
%   - earlier diagnostics showed poor RT sensitivity and poor RT-based fit
%   - the choice-only inversion is being debugged first
if M.rt == 1

    % Reconstruct model-predicted RT proxy across the full 6-block schedule
    Rpred = [];
    for b = 1:numel(MDPfit)
        Rb = MDP_Flanker_RT(MDPfit{b});
        Rpred = [Rpred, Rb];
    end

    % Observed and predicted RT vectors may occasionally differ slightly in
    % length, so compare only over the shared overlap.
    yr = Y.r(:);
    rp = Rpred(:);
    nUse = min(numel(yr), numel(rp));

    % Gaussian likelihood on log-RT
    L = L + sum(log(spm_Npdf(log(yr(1:nUse)), rp(1:nUse) - log(2), 1/256)));
end
end