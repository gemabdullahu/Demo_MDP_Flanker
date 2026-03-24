function [mdp, Z] = FLK_Online_3(p)

% This routine uses a deep temporal partially observed Markov decision
% process to simulate performance on a block-based Flanker task. The model
% follows the general logic of DEMO_MDP_Stroop, but adapts it to a Flanker
% setting in which synthetic participants must select the direction of a
% central target while overcoming interference from surrounding flankers.
%
% In this formulation, cognitive effort is expressed as the need to
% resolve conflict between target-relevant and flanker-driven response
% tendencies, under different contextual regimes of congruency. Unlike the
% Stroop task, where conflict is defined by competition between word and
% colour dimensions, here conflict is defined by competition between the
% target and flanker directions.
%
% The model is structured as a deep (hierarchical) MDP: a slower level
% encodes block-wise contextual regimes that determine the expected
% congruency statistics of trials (e.g., mostly congruent, mostly
% incongruent, non-predictive), while a faster within-trial level governs
% stimulus processing, response selection, and outcome evaluation. This
% enables context-dependent control demands to be represented in a way
% analogous to the higher-level structure in DEMO_MDP_Stroop.
%
% Importantly, this model is motivated by behavioural data from a
% block-based Flanker experiment in which trials are organised into blocks
% with different congruency statistics. This empirical structure justifies
% the inclusion of an explicit context factor and the use of block-wise
% schedules, because the task environment changes systematically across
% blocks and can be modelled as context-dependent stimulus statistics.
%
% In addition, this routine is intended to support simulation, fitting,
% and recovery analyses using behavioural summaries (e.g., accuracy and
% reaction-time-like measures), in the spirit of DEMO_MDP_Stroop. The goal
% is not only to simulate Flanker performance, but also to test whether
% latent parameters governing prior biases and control-related preferences
% can be identified from synthetic or empirical block-based Flanker data.
% -------------------------------------------------------------------------
% Notes for users / current implementation limits
% -------------------------------------------------------------------------
% This model was developed to mirror the hierarchical logic of
% DEMO_MDP_Stroop, but several non-trivial issues arose when adapting that
% framework to a block-based Flanker task. These are documented here so
% that future users can understand the current implementation and, if
% desired, refine it further.
%
% 1) Self-generated response outcomes were turned off deliberately
% ----------------------------------------------------------------
% In the original Stroop demo, the verbal response modality is treated as a
% self-generated outcome. We initially tried to preserve that feature here.
% However, in the current Flanker response formulation this caused clear
% corruption of the response channel.
%
% Specifically:
%   - with the original 3-outcome Flanker response modality, self-generated
%     handling produced spurious null outcomes
%   - when we padded the response modality to 5 outcomes (to imitate the
%     Stroop response format more closely), probability mass leaked into
%     dummy/null categories that were not meaningful responses
%   - turning off self-generated response outcomes removed this corruption
%
% For that reason, the current implementation does NOT use Stroop-style
% self-generated response handling for modality 4. This is a practical
% engineering decision, not a claim that self-generated outcomes are
% theoretically inappropriate in general. Rather, in this specific Flanker
% implementation they did not interact cleanly with the reduced response
% coding.
%
% 2) Lambda did not behave as a useful noise parameter once self-generated
%    outcomes were turned off
% -------------------------------------------------------------------------
% After disabling self-generated response outcomes, changing lambda no
% longer had meaningful effects on the behavioral summaries of interest.
% This suggested that lambda was no longer the effective "noise knob" for
% the current response architecture.
%
% Instead, the useful way to desaturate responses was to add explicit motor
% noise / response slip directly to the fast-level keypress likelihood
% A1{4}. This is controlled by the parameter eta in the current code.
%
% In practice, eta became the parameter that controls how deterministic the
% final keypress mapping is:
%   - very small eta -> nearly deterministic responses
%   - larger eta     -> softer response mapping
%
% 3) RT proxy magnitude was often too small
% -----------------------------------------
% RT-like summaries in this model are derived from predictive entropy over
% the response modality, following the general Stroop approach. However, in
% the Flanker model the RT proxy often showed a much smaller dynamic range
% across the (c,e) parameter grid than expected.
%
% Diagnostics suggested that:
%   - the raw RT proxy often varied only weakly across parameter settings
%   - after transforming to an observed RT scale, the RT surface became
%     even flatter
%   - the high-entropy fraction could become nearly invariant across the
%     parameter grid
%
% Eta contributed strongly to this behaviour. When eta was very small, the
% response channel became almost deterministic on congruent/easy trials,
% compressing the low-entropy state. Increasing eta reduced this
% saturation, but introduced a tradeoff: values that softened the response
% mapping enough to help trialwise choice identifiability could also change
% the RT geometry.
%
% Therefore, RT-based recovery in this Flanker implementation should be
% treated cautiously.
%
% 4) Full inversion remains unresolved
% ------------------------------------
% Summary-level analyses show meaningful dependence on c and e, and the
% surrogate recovery analyses remain useful within that scope. However, a
% fully validated full inversion has not yet been achieved for the Flanker
% model.
%
% Several concrete issues were identified during development:
%
% (a) Template mismatch:
%     Early inversion attempts reused already solved MDP objects as the
%     inversion template. This prevented parameter manipulations from being
%     expressed cleanly in regenerated behavior. The inversion pathway was
%     therefore rebuilt around a clean unsolved 6-block template.
%
% (b) Observation-assignment mismatch:
%     The original Stroop-style outcome-assignment logic did not transfer
%     directly to the Flanker schedule. The code was updated so that
%     observed outcomes are now passed block-by-block as full outcome
%     matrices.
%
% (c) Generator parameterization mismatch:
%     Early generator versions used p.c and p.e as absolute strengths,
%     whereas the synthetic Flanker datasets had been generated as local
%     offsets around calibrated baselines. The generator was updated to
%     match the simulation parameterization.
%
% (d) RT pathway weakness:
%     The RT proxy was found to be weak and nearly flat over the local
%     parameter grid, so RT is currently excluded from full inversion.
%
% Even after these fixes, the choice-likelihood pathway still does not
% reliably favor the true generating parameter values. Thus, the remaining
% failure of full inversion cannot currently be attributed to
% identifiability alone.
%
% In short:
%   - the current model is useful for simulation and summary-level analyses
%   - surrogate recovery remains meaningful within that scope
%   - the present full inversion is still exploratory and not yet validated
%
% 5) Main directions for improvement
% ----------------------------------
% Future researchers who wish to improve this model may want to explore:
%
%   (a) a revised choice-likelihood pathway, since the main remaining issue
%       is that full inversion still does not reliably favor the true
%       generating parameter values
%   (b) alternative likelihood formulations that preserve more of the
%       parameter sensitivity present in internal response beliefs
%   (c) a redesigned response coding scheme that would allow self-generated
%       outcomes to be reintroduced without corrupting the response channel,
%       so that lambda could be used again as a meaningful precision term
%       rather than relying primarily on explicit response noise via eta
%   (d) a more systematic analysis of the tradeoff between eta, RT
%       geometry, and trialwise choice identifiability
%
% -------------------------------------------------------------------------

if nargin < 1
    p = struct();
end

% -------------------------
% Parameters and options
% -------------------------
if isfield(p,'pc'),   pc = p.pc;   else, pc = [0.75 0.25 0.50]; end
try e = exp(p.e);                  catch, e = 0.051366;         end   % tendency to follow flankers
try c = exp(p.c);                  catch, c = 0.22;             end   % preference for being correct
if isfield(p,'tau'),  tau = p.tau; else, tau = 8;              end
if isfield(p,'eta'),  eta = p.eta; else, eta = 0.10;           end   % response noise in the motor channel

do_run = true;
if isfield(p,'run')
    do_run = p.run;
end

%% =========================================================================
% FAST level: within-trial Flanker dynamics
% =========================================================================
% f1 TargetDir    : left / right
% f2 Congruency   : congruent / incongruent
% f3 TaskSequence : cue / null / response
% f4 Instruction  : attend target / attend flanker
% f5 Response     : report target / report flanker
% f6 Correct?     : correct / incorrect
label1.factor = {'TargetDir','Congruency','TaskSequence','Instruction','Response','Correct?'};

% Priors over initial hidden states
D1 = cell(1,6);
D1{1} = ones(2,1);
D1{2} = ones(2,1);
D1{3} = ones(3,1);
D1{4} = ones(2,1);
D1{5} = ones(2,1);
D1{6} = ones(2,1);

% State transitions within a trial
B1 = cell(1,6);
B1{1} = eye(2);     % target direction fixed within trial
B1{2} = eye(2);     % congruency fixed within trial
B1{3} = [1 0 0; ...
         0 0 0; ...
         0 1 1];    % cue -> null -> response
B1{4} = eye(2);     % instruction fixed
B1{5} = eye(2);     % response factor fixed
B1{5}(:,:,2) = eye(2);
B1{6} = eye(2);     % correctness fixed

% Likelihoods / observations
% o1 flanker stimulus : L / R / null
% o2 target stimulus  : L / R / null
% o3 cue              : AT / AF / none
% o4 keypress         : L / R / null
A1 = cell(1,4);
A1{1} = zeros(3,2,2,3,2,2,2);
A1{2} = zeros(3,2,2,3,2,2,2);
A1{3} = zeros(3,2,2,3,2,2,2);
A1{4} = zeros(3,2,2,3,2,2,2);

for f1 = 1:2
for f2 = 1:2
for f3 = 1:3
for f4 = 1:2
for f5 = 1:2
for f6 = 1:2

    % Flanker direction is determined by target direction and congruency
    if f2 == 1
        fl = f1;
    else
        fl = 3 - f1;
    end

    % Stimulus modalities
    if f3 == 1
        % Cue phase: no target or flanker shown yet
        A1{1}(3,  f1,f2,f3,f4,f5,f6) = 1;
        A1{2}(3,  f1,f2,f3,f4,f5,f6) = 1;
        A1{3}(f4, f1,f2,f3,f4,f5,f6) = 1;
    else
        % Null / response phase: stimuli visible, cue off
        A1{1}(fl, f1,f2,f3,f4,f5,f6) = 1;
        A1{2}(f1, f1,f2,f3,f4,f5,f6) = 1;
        A1{3}(3,  f1,f2,f3,f4,f5,f6) = 1;
    end

    % Keypress modality
    % If reporting target, motor output follows target direction.
    % If reporting flanker, motor output follows flanker direction.
    if f3 == 3
        if f5 == 1
            A1{4}(:, f1,f2,f3,f4,f5,f6) = 0;
            A1{4}(f1,   f1,f2,f3,f4,f5,f6) = 1 - eta;
            A1{4}(3-f1, f1,f2,f3,f4,f5,f6) = eta;
        else
            A1{4}(:, f1,f2,f3,f4,f5,f6) = 0;
            A1{4}(fl,   f1,f2,f3,f4,f5,f6) = 1 - eta;
            A1{4}(3-fl, f1,f2,f3,f4,f5,f6) = eta;
        end
    else
        A1{4}(:, f1,f2,f3,f4,f5,f6) = 0;
        A1{4}(3, f1,f2,f3,f4,f5,f6) = 1;
    end

end
end
end
end
end
end

% Preferences at the fast level are neutral
Cfast = cell(1,4);
for m = 1:4
    Cfast{m} = zeros(size(A1{m},1),1);
end

% Response modality is NOT treated as self-generated in this implementation
nfast = zeros(numel(A1),2);
nfast(4,:) = 0;

% Compile fast model
FAST = struct();
FAST.T      = 2;
FAST.A      = A1;
FAST.B      = B1;
FAST.C      = Cfast;
FAST.D      = D1;
FAST.n      = nfast;
FAST.tau    = tau;
FAST.chi    = -exp(64);
FAST.label  = label1;
FAST        = spm_MDP_check(FAST);

%% =========================================================================
% SLOW level: block/context structure
% =========================================================================
% Narrative / Context / Instruction / Response
label2.factor = {'Narrative','Context','Instruction','Response'};

D2 = cell(1,4);
D2{1} = [1;0];      % cue phase at trial start
D2{2} = ones(3,1);  % contexts: MC / MI / NP
D2{3} = [1;1];      % instruction prior
D2{4} = [1;1];      % response prior

B2 = cell(1,4);
B2{1} = [0 0; 1 1];   % cue -> response
B2{2} = eye(3);       % context fixed within block
B2{3} = eye(2);       % instruction fixed
B2{4} = zeros(2,2,2);
B2{4}(1,:,2) = 1;     % choose target response
B2{4}(2,:,1) = 1;     % choose flanker response

% Slow-level likelihoods
% 1 predicts FAST TaskSequence
% 2 predicts FAST Instruction
% 3 predicts FAST Response
% 4 predicts FAST Correct?
% 5 effort channel
% 6 predicts FAST Congruency
A2 = cell(1,6);
A2{1} = zeros(3,2,3,2,2);
A2{2} = zeros(2,2,3,2,2);
A2{3} = zeros(2,2,3,2,2);
A2{4} = zeros(2,2,3,2,2);
A2{5} = zeros(2,2,3,2,2);
A2{6} = zeros(2,2,3,2,2);

for f1 = 1:2
for ctx = 1:3
for ins = 1:2
for rsp = 1:2

    % Predicted task sequence
    if f1 == 1
        A2{1}(1,f1,ctx,ins,rsp) = 1;   % cue
    else
        A2{1}(2,f1,ctx,ins,rsp) = 1;   % response
    end
    A2{1}(3,f1,ctx,ins,rsp) = 0;

    % Predicted instruction
    A2{2}(ins,f1,ctx,ins,rsp) = 1;

    % Predicted response mode
    A2{3}(rsp,f1,ctx,ins,rsp) = 1;

    % Predicted correctness:
    % reporting target is always correct;
    % reporting flanker is only correct on congruent trials
    if rsp == 1
        A2{4}(1,f1,ctx,ins,rsp) = 1;
    else
        A2{4}(1,f1,ctx,ins,rsp) = pc(ctx);
        A2{4}(2,f1,ctx,ins,rsp) = 1 - pc(ctx);
    end

    % Effort channel left neutral
    A2{5}(:,f1,ctx,ins,rsp) = [0;0];

    % Predicted congruency statistics for the current context
    A2{6}(1,f1,ctx,ins,rsp) = pc(ctx);
    A2{6}(2,f1,ctx,ins,rsp) = 1 - pc(ctx);

end
end
end
end

C2 = cell(1,6);
for m = 1:6
    C2{m} = zeros(size(A2{m},1),1);
end
C2{4} = [c; -c];   % preference for correct over incorrect

% Prior over response policies (Stroop-style)
E2 = spm_softmax([+e; -e]);

% Compile deep model
mdp = struct();
mdp.MDP   = FAST;
mdp.A     = A2;
mdp.B     = B2;
mdp.C     = C2;
mdp.D     = D2;
mdp.E     = E2;
mdp.T     = 2;
mdp.tau   = tau;
mdp.label = label2;

% Deep links from slow to fast level
mdp.link = zeros(numel(mdp.MDP.D), numel(mdp.A));
mdp.link(3,1) = 1;   % FAST TaskSequence <- A2{1}
mdp.link(4,2) = 1;   % FAST Instruction  <- A2{2}
mdp.link(5,3) = 1;   % FAST Response     <- A2{3}
mdp.link(6,4) = 1;   % FAST Correct?     <- A2{4}
mdp.link(2,6) = 1;   % FAST Congruency   <- A2{6}

mdp = spm_MDP_check(mdp);

if ~do_run
    Z = [];
    fprintf('[FLK_Online_3] returning template only\n');
    return
end

OPTIONS = struct('gamma',1,'plot',0);
Z = spm_MDP_VB_X(mdp, OPTIONS);





