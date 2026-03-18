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
% 4) Full trialwise inversion remains unresolved
% ----------------------------------------------
% Summary-level analyses (especially for accuracy) show meaningful and
% interpretable dependence on c and e, and surrogate recovery analyses work
% reasonably well. However, fully satisfactory trialwise inversion has not
% yet been achieved.
%
% Two major issues were identified:
%
% (a) Schedule replay:
%     Early inversion attempts evaluated candidate parameter settings on
%     regenerated trial sequences that did not match the realized trial
%     sequence in the synthetic dataset. A forced-schedule implementation
%     was introduced to replay target direction, congruency, and
%     instruction/block context more faithfully.
%
% (b) Weak trialwise choice identifiability:
%     Even after fixing schedule alignment, the trialwise likelihood
%     remained weak. The main problem is that parameter changes can be seen
%     in internal response beliefs, but those differences are often too
%     small after the final response mapping. Congruent trials tend to be
%     almost deterministic, while incongruent trials carry most of the
%     useful signal. As a result, likelihood surfaces remain shallow and
%     full recovery is unstable.
%
% In short:
%   - the current model is useful for simulation and summary-level analyses
%   - the response channel is stable only when self-generated outcomes are
%     disabled and explicit response noise is handled via eta
%   - full trialwise inversion remains an open problem for future work
%
% 5) Main directions for improvement
% ----------------------------------
% Future researchers who wish to improve this model may want to explore:
%
%   (a) a cleaner replay-ready inversion setup, closer in spirit to the
%       original Stroop demo
%   (b) alternative likelihood formulations that preserve more of the
%       parameter sensitivity present in internal response beliefs
%   (c) a redesigned response coding that would allow self-generated
%       outcomes to be reintroduced without corrupting the response channel
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



%% =========================================================
% Flanker analogue of Stroop "Congruency x task condition"
% Behavioral sanity check before surrogate recovery
% =========================================================

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
subplot(3,1,2); hold on

bins = 0.4:0.02:1.2;

histogram(exp(H_MC(3,H_MC(1,:)==1) + randn(size(H_MC(3,H_MC(1,:)==1)))/16)/2, bins)
histogram(exp(H_MC(3,H_MC(1,:)==0) + randn(size(H_MC(3,H_MC(1,:)==0)))/16)/2, bins)

histogram(exp(H_MI(3,H_MI(1,:)==1) + randn(size(H_MI(3,H_MI(1,:)==1)))/16)/2, bins)
histogram(exp(H_MI(3,H_MI(1,:)==0) + randn(size(H_MI(3,H_MI(1,:)==0)))/16)/2, bins)

histogram(exp(H_NP(3,H_NP(1,:)==1) + randn(size(H_NP(3,H_NP(1,:)==1)))/16)/2, bins)
histogram(exp(H_NP(3,H_NP(1,:)==0) + randn(size(H_NP(3,H_NP(1,:)==0)))/16)/2, bins)

axis square
xlabel('Reaction Time (s)')
ylabel('Counts')
legend('MC cong','MC incong', ...
       'MI cong','MI incong', ...
       'NP cong','NP incong', ...
       'Location','best')
title('Reaction time distribution')


figure(1); clf;
subplot(3,1,1)
xpos = 1:numel(B);
bar(xpos, (B(:)')*100, 'EdgeColor',[1 1 1])
set(gca,'XTick',xpos,'XTickLabel',labels);
xtickangle(45);
axis square;
title('Percentage correct');
ylabel('% correct')

%% =========================================================
% FLK surrogate recovery
% Mirrors DEMO_MDP_Stroop surrogate logic as closely as possible
% =========================================================

rng default

[x, y] = meshgrid((-1:1/2:1)/4, (-1:1/2:1)/4);
X = [x(:), y(:)];  

c0 = 0.22;
e0 = 0.051366;

ctx_blocks = [1 2 3 2 1 3];
T_blocks   = [60 60 30 60 60 30];

for i = 1:size(X,1)

    p = struct();
    p.c = log(c0) + X(i,1);
    p.e = log(e0) + X(i,2);
    p.pc  = [0.75 0.25 0.50];
    p.run = false;
    p.tau = 8;
    p.eta = 0.010;   

    mdp0 = FLK_Online_3(p);

    rng default  

    for b = 1:numel(ctx_blocks)

        mdp = mdp0;
        mdp.T = T_blocks(b);

        if ctx_blocks(b) == 1
            mdp.D{2} = [1;0;0];   % MC
        elseif ctx_blocks(b) == 2
            mdp.D{2} = [0;1;0];   % MI
        else
            mdp.D{2} = [0;0;1];   % NP
        end

        MDP(i,b) = spm_MDP_VB_X(mdp, OPTIONS);

        tr = MDP(i,b).mdp(2:end);
        oF = arrayfun(@(x) x.o(1,2), tr);
        oT = arrayfun(@(x) x.o(2,2), tr);
        oR = arrayfun(@(x) x.o(4,2), tr);

        validStim = (oF~=3) & (oT~=3);

    end
end

% =========================================================
% FLANKER PERFORMANCE SUMMARIES 
% Full 6-block schedule, Stroop-style pooled summaries
% =========================================================

nRT = 0;
for b = 1:6
    nRT = nRT + (MDP(1,b).T - 1);
end

B = zeros(1, size(X,1));
R = nan(nRT, size(X,1));

for k = 1:size(MDP,1)

    acc_sum = 0;
    rt_all = [];

    for b = 1:6
        [stim{k,b}, resp{k,b}] = MDP_Flanker_SR(MDP(k,b));

        H = zeros(2, numel(resp{k,b}));
        for i = 1:numel(resp{k,b})
            H(1,i) = strcmp(stim{k,b}.Tdir{i}, stim{k,b}.Fdir{i});
            H(2,i) = strcmp(['"' stim{k,b}.Tdir{i} '"'], resp{k,b}{i});
        end

        block_acc = mean(H(2,:));
        acc_sum = acc_sum + block_acc;

        rt = MDP_Flanker_RT(MDP(k,b));
        rt = exp(rt(:) + randn(size(rt(:)))/16)/2;

        rt_all = [rt_all; rt];

    end

    B(k) = acc_sum / 6;
    R(1:numel(rt_all), k) = rt_all;

end

B = B(:);
Rmean = mean(R,1,'omitnan')';

%% ---------------------------------------------------------
% Surrogate fit to accuracy, Stroop-style polynomial
% ----------------------------------------------------------
U = [ones(size(X,1),1), X, X(:,1).*X(:,2), X(:,1).^2, X(:,2).^2];

% Accuracy surrogate
M = struct();
M.L       = @(P,M,U,Y) sum(log(spm_Npdf(Y, exp(U*P.beta)./(exp(U*P.beta)+exp(1)), exp(-P.pi))));
M.pE.beta = zeros(size(U,2),1);
M.pE.pi   = 4;
M.pC      = eye(spm_length(M.pE));

[Ep_acc,Cp_acc,F_acc] = spm_nlsi_Newton(M,U,B);

DCM.Ep = Ep_acc;
DCM.Cp = Cp_acc;
DCM.F  = F_acc;
DCM.M  = M;
[PCM_acc,~,BMA_acc] = spm_dcm_bmr_all(DCM,'beta');

Bp  = BMA_acc.Ep.beta;
BPp = PCM_acc.Pp.beta;

% RT surrogate
M.L       = @(P,M,U,Y) sum(log(spm_Npdf(log(Y), U*P.beta, exp(-P.pi))));
[Ep_rt,Cp_rt,F_rt] = spm_nlsi_Newton(M,U,Rmean);

DCM.Ep = Ep_rt;
DCM.Cp = Cp_rt;
DCM.F  = F_rt;
DCM.M  = M;
[PCM_rt,~,BMA_rt] = spm_dcm_bmr_all(DCM,'beta');

Rp  = BMA_rt.Ep.beta;
RPp = PCM_rt.Pp.beta;

% Plotting 

spm_figure('GetWin','Figure 8'); clf

subplot(3,2,1)
imagesc(X(:,1),X(:,2),100*reshape(B,[sqrt(length(B)),sqrt(length(B))]))
title('Percentage correct')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')

subplot(3,2,2)
imagesc(X(:,1),X(:,2),reshape(Rmean,[sqrt(length(Rmean)),sqrt(length(Rmean))]))
title('Mean reaction time')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')

subplot(3,2,3)
imagesc(X(:,1),X(:,2),100*reshape(exp(U*Bp)./(exp(U*Bp)+exp(1)),[sqrt(length(B)),sqrt(length(B))]))
title('Percentage correct (fit)')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')
caxis([min(B*100),max(B*100)])

subplot(3,2,4)
imagesc(X(:,1),X(:,2),reshape(exp(U*Rp),[sqrt(length(Rmean)),sqrt(length(Rmean))]))
title('Mean reaction time (fit)')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')
caxis([min(Rmean),max(Rmean)])

subplot(3,2,5)
bar(BPp,'FaceColor',[.7 .7 .9],'EdgeColor',[1 1 1])
axis square
title('Posterior probabilities (% correct)')
xlabel('Parameter')
ylabel('Probability')

subplot(3,2,6)
bar(RPp,'FaceColor',[.7 .7 .9],'EdgeColor',[1 1 1])
axis square
title('Posterior probabilities (reaction time)')
xlabel('Parameter')
ylabel('Probability')

%--------------------------------------------------------------------------
% Model fits
%--------------------------------------------------------------------------

clear M
U  = [];

M.L       = @MDP_Flanker_L;
M.G       = @(p,Y) MDP_Flanker_Gen(p,Y);
M.pE.c    = log(0.22);
M.pE.e    = log(0.051366);
M.pC      = eye(spm_length(M.pE))/256;
M.ch      = 1;
M.rt      = 0;
M.verbose = 0;                

nFit = 4;   
nFit = min(nFit, size(MDP,1));

pool = gcp('nocreate');
if isempty(pool)
    parpool(14);
elseif pool.NumWorkers ~= 14
    delete(pool);
    parpool(14);
end

Yall = cell(nFit,1);

for i = 1:nFit
    Y.o   = {};
    Y.sch = {};
    obs_idx = 0;
    
    for b = 1:6
        for j = 2:length(MDP(i,b).mdp)
            obs_idx = obs_idx + 1;
    

            Y.o{obs_idx} = MDP(i,b).mdp(j).o;
   
            Y.sch{obs_idx} = struct();
            Y.sch{obs_idx}.block = b;
            Y.sch{obs_idx}.Tdir  = MDP(i,b).mdp(j).s(1,2);  % TargetDir
            Y.sch{obs_idx}.Cong  = MDP(i,b).mdp(j).s(2,2);  % Congruency
            Y.sch{obs_idx}.Instr = MDP(i,b).mdp(j).s(4,2);  % Instruction
        end
    end
    
    Y.r = [];
    Yall{i} = Y;
end

Ep = nan(nFit, 2);
Cp = nan(nFit, 3);   % [Var(c) Var(e) Cov(c,e)]

parfor i = 1:nFit
    Y = Yall{i};

    [EP,CP,~] = spm_nlsi_Newton(M, U, Y);

    Ep(i,:) = spm_vec(EP)';
    Cp(i,:) = [diag(CP)' CP(1,2)];

    fprintf('Finished inversion %d/%d\n', i, nFit);
end


spm_figure('GetWin','Figure 5'); clf

subplot(3,1,1)
bar(1:nFit, Ep(:,1), 'FaceColor',[.7 .7 .9], 'EdgeColor',[1 1 1]), hold on
errorbar(1:nFit, Ep(:,1), ...
    1.65*sqrt(Cp(:,1)), -1.65*sqrt(Cp(:,1)), ...
    'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
plot(1:nFit, X(1:nFit,1), '.r', 'MarkerSize', 20)
hold off
xlabel('Dataset')
ylabel('Parameter estimate')
title('Demand / correctness prior (c)')

subplot(3,1,2)
bar(1:nFit, Ep(:,2), 'FaceColor',[.7 .7 .9], 'EdgeColor',[1 1 1]), hold on
errorbar(1:nFit, Ep(:,2), ...
    1.65*sqrt(Cp(:,2)), -1.65*sqrt(Cp(:,2)), ...
    'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
plot(1:nFit, X(1:nFit,2), '.r', 'MarkerSize', 20)
hold off
xlabel('Dataset')
ylabel('Parameter estimate')
title('Effort / habit prior (e)')

subplot(3,1,3)
bar(1:nFit, Ep(:,2)-Ep(:,1), 'FaceColor',[.7 .7 .9], 'EdgeColor',[1 1 1]), hold on
errorbar(1:nFit, Ep(:,2)-Ep(:,1), ...
    1.65*sqrt(Cp(:,1)+Cp(:,2)-2*Cp(:,3)), ...
   -1.65*sqrt(Cp(:,1)+Cp(:,2)-2*Cp(:,3)), ...
    'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
plot(1:nFit, X(1:nFit,2)-X(1:nFit,1), '.r', 'MarkerSize', 20)
hold off
xlabel('Dataset')
ylabel('Parameters')
title('Differences (e - c)')

Zfit.Ep = Ep;
Zfit.Cp = Cp;
Zfit.nFit = nFit;
save('Zfit_flanker.mat', 'Zfit')

%--------------------------------------------------------------------------
% Local Functions
%--------------------------------------------------------------------------

function [rt_end, rt_meanv] = MDP_Stroop_RT(MDP, doDebug)

if nargin < 2
    doDebug = false;
end

nTrials = length(MDP.mdp) - 1;

rt_end   = nan(1, nTrials);
rt_meanv = nan(1, nTrials);

for i = 2:length(MDP.mdp)
    x = [];

    for k = 1:size(MDP.mdp(i).xn{1},1)
        xn = cell(1, numel(MDP.mdp(i).xn));
        for j = 1:numel(MDP.mdp(i).xn)
            xn{j} = MDP.mdp(i).xn{j}(k,:,2,1);
        end
        x(:,end+1) = spm_dot(MDP.mdp(i).A{4}, xn);
    end

    v = -diag(x' * spm_log(x));

    rt_end(i-1)   = v(end);
    rt_meanv(i-1) = mean(v);

    if doDebug && i <= 6
        fprintf('trial %d | v(end)=%.6f | mean(v)=%.6f\n', ...
            i-1, v(end), mean(v));
        fprintf('  x(:,end) = ');
        disp(x(:,end)')
        fprintf('  full v   = ');
        disp(v(:)')
    end
end

if doDebug
    fprintf('\n===== STROOP RT SUMMARY =====\n');
    fprintf('rt_end   range = [%.6f %.6f], std = %.6f\n', ...
        min(rt_end), max(rt_end), std(rt_end));
    fprintf('rt_meanv range = [%.6f %.6f], std = %.6f\n', ...
        min(rt_meanv), max(rt_meanv), std(rt_meanv));
end
end

function [stim, resp] = MDP_Flanker_SR(MDP)
% This routine reports stimuli and response sequences without generating an
% animation

str = {'L','R',' '};
dir = {'L','R'};

if isfield(MDP,'mdp')
    stim.Fdir = [];
    stim.Tdir = [];
    resp      = [];
    for i = 1:length(MDP.mdp)
        [s, r] = MDP_Flanker_SR(MDP.mdp(i));
        stim.Fdir = [stim.Fdir, s.Fdir];
        stim.Tdir = [stim.Tdir, s.Tdir];
        resp      = [resp, r];
    end
else
    stim.Fdir = [];
    stim.Tdir = [];
    resp      = [];
    for t = 1:MDP.T
        if MDP.o(3,t) == 3
            if t == 2
                stim.Tdir = [stim.Tdir, {dir{MDP.o(2,t)}}];
                stim.Fdir = [stim.Fdir, str(MDP.o(1,t))];
            end
        end
        if t == 2 && ~isempty(stim.Tdir)
            resp = [resp, {['"' str{MDP.o(4,t)} '"']}];
        end
    end
end
end

function L = MDP_Flanker_L(P,M,~,Y)

L = 0;

MDP = M.G(P, Y);

obs_idx = 0;
nMismatch = 0;

for b = 1:numel(MDP)
    for i = 2:length(MDP(b).mdp)

        obs_idx = obs_idx + 1;

        oi = Y.o{obs_idx};
        sch = Y.sch{obs_idx};

        % observed response at response time
        obs_resp = oi(4,2);

        % generated realized trial identity
        gen_Tdir = MDP(b).mdp(i).s(1,2);
        gen_Cong = MDP(b).mdp(i).s(2,2);
        gen_Instr = MDP(b).mdp(i).s(4,2);

        % check alignment
        if gen_Tdir ~= sch.Tdir || gen_Cong ~= sch.Cong || gen_Instr ~= sch.Instr
            nMismatch = nMismatch + 1;
        end

        % use final posterior state beliefs, same spirit as Stroop
        xn = cell(1, numel(MDP(b).mdp(i).xn));
        for j = 1:numel(MDP(b).mdp(i).xn)
            xn{j} = MDP(b).mdp(i).xn{j}(end,:,2,1);
        end

        x = spm_softmax(spm_log(spm_dot(MDP(b).mdp(i).A{4}, xn)));

        L = L + spm_log(x(obs_resp));
    end
end

end

function MDPgen = MDP_Flanker_Gen(p, Y)

pc_fix  = [0.75 0.25 0.50];
tau_fix = 8;
eta_fix = 0.10;

ctx_blocks = [1 2 3 2 1 3];
T_blocks   = [60 60 30 60 60 30];

pp = struct();
pp.run = false;
pp.pc  = pc_fix;
pp.c   = p.c;
pp.e   = p.e;
pp.tau = tau_fix;
pp.eta = eta_fix;

mdp0 = FLK_Online_3(pp);

OPTIONS.gamma = 1;

obs_idx = 0;

for b = 1:numel(ctx_blocks)
    mdp = mdp0;
    mdp.T = T_blocks(b);

    if ctx_blocks(b) == 1
        mdp.D{2} = [1;0;0];
    elseif ctx_blocks(b) == 2
        mdp.D{2} = [0;1;0];
    else
        mdp.D{2} = [0;0;1];
    end

    rng default
    Zb = spm_MDP_VB_X(mdp, OPTIONS);

    % attach observed schedule for later trial matching if available
    if nargin > 1 && isfield(Y,'sch') && ~isempty(Y.sch)
        for j = 2:length(Zb.mdp)
            obs_idx = obs_idx + 1;
            Zb.mdp(j).sch = Y.sch{obs_idx};
        end
    end

    if b == 1
        MDPgen = repmat(Zb, 1, numel(ctx_blocks));
    end
    MDPgen(b) = Zb;
end
end

end

