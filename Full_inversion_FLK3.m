%% =========================================================
% PART 1. Exploratory full inversion: Flanker adaptation of Stroop fitting
% =========================================================
% This section was inspired directly by the full-inversion stage in
% `DEMO_MDP_Stroop`, where synthetic datasets are inverted in order to
% recover latent prior-belief parameters from simulated behavior.
%
% However, the Flanker model differs from Stroop in several important ways,
% so the original Stroop inversion code could not be transferred without
% modification.
%
% -------------------------------------------------------------------------
% Why the Stroop inversion had to be changed
% -------------------------------------------------------------------------
% The original Stroop implementation assumes:
%
%   1. two task conditions that can be concatenated trial-by-trial
%   2. a generator operating on a single Stroop-style template
%   3. a likelihood function that assigns observations through the Stroop
%      data structure
%
% In the present Flanker model, none of these assumptions holds exactly.
% Instead, the model is organized as a 6-block schedule:
%
%   [MC, MI, NP, MI, MC, NP]
%
% with unequal block lengths. Therefore, the Stroop logic had to be adapted
% so that inversion operates on the full Flanker schedule rather than on
% two matched conditions.
%
% -------------------------------------------------------------------------
% Main modifications made here
% -------------------------------------------------------------------------
% 1. Clean unsolved Flanker template
%    The original Stroop inversion uses a Stroop template directly inside
%    the generator. In the Flanker case, using an already solved `MDP`
%    object as the inversion template produced pathological behavior,
%    because edits to latent parameters were not being expressed cleanly in
%    regenerated behavior. To address this, we now build a fresh unsolved
%    6-block Flanker template (`FLK_TEMPLATE`) directly from
%    `FLK_Online_3(..., run=false)` before inversion.
%
% 2. Full 6-block schedule rather than 2-condition concatenation
%    Stroop combines two conditions trial-by-trial. In the Flanker task,
%    the inversion target is a blockwise schedule, so the observed data are
%    now passed block-by-block:
%
%       Y.o{b} = observed 6 x T outcome matrix for block b
%
%    rather than as a per-trial concatenation of two conditions.
%
% 3. Flanker-specific generator
%    The generator was rewritten so that candidate latent parameters are
%    applied to the clean 6-block Flanker template, rather than to the
%    Stroop-style single-template structure. It also now uses the same
%    baseline-centered parameterization as the synthetic data generation:
%
%       c = exp(log(c0) + p.c)
%       e = exp(log(e0) + p.e)
%
%    This was necessary because the synthetic datasets were generated as
%    local offsets around calibrated baseline values, not as absolute
%    strengths.
%
% 4. Blockwise likelihood assignment
%    The likelihood function now assigns observed outcomes to each block as
%    full block-level outcome matrices before calling `spm_MDP_VB_X`. This
%    replaces earlier trialwise assignment attempts that were incompatible
%    with the clean unsolved Flanker template.
%
% 5. RT temporarily disabled
%    The original Stroop full inversion includes both choice and RT data.
%    In the present Flanker implementation, the RT proxy was found to be
%    weak, nearly flat over the local parameter grid, and poorly captured by
%    the pooled surrogate fit. For this reason, RT is currently switched off
%    in the full inversion (`M.rt = 0`) so that the inversion can be
%    inspected first under choice data alone.
%
% -------------------------------------------------------------------------
% Problems encountered during adaptation
% -------------------------------------------------------------------------
% Several structural problems were identified while adapting the Stroop
% inversion to Flanker:
%
%   - Using `MDP(1,:)` from already solved simulations as the inversion
%     template caused parameter manipulations to have little or no effect on
%     regenerated behavior.
%
%   - The original outcome-assignment logic from Stroop did not match the
%     Flanker data structure, because the Flanker model is block-based and
%     not organized as two directly concatenated task conditions.
%
%   - The RT likelihood initially failed because the predicted RT vector and
%     observed RT vector were not aligned in size; later diagnostics also
%     showed that the Flanker RT proxy itself is weakly informative.
%
%   - Early generator versions used a parameterization inconsistent with the
%     simulation code, treating `p.c` and `p.e` as absolute strengths rather
%     than offsets around calibrated baselines.
%
%   - Additional diagnostics showed that the clean forward generator does
%     express `c` behaviorally, but the full choice likelihood still does
%     not reliably favor the true generating parameter values.
%
% -------------------------------------------------------------------------
% Current status
% -------------------------------------------------------------------------
% The present code should therefore be understood as an exploratory
% Flanker-homologous adaptation of the Stroop full inversion, not yet as a
% validated recovery procedure.
%
% In particular, several checks now pass:
%
%   - a clean unsolved schedule template is used
%   - the generator is parameterized consistently with the simulation code
%   - observed outcomes are assigned blockwise in a structurally appropriate
%     way
%   - the generator does show behavioral sensitivity to changes in `c`
%
% However, important problems remain:
%
%   - profile likelihood checks do not yet favor the true generating value
%     of `c`
%   - blockwise choice-likelihood profiles still prefer larger positive `c`
%     values over the true generating offset
%   - forward pooled behavior and choice likelihood are not yet cleanly
%     aligned as functions of `c`
%
% -------------------------------------------------------------------------
% What still needs to be fixed
% -------------------------------------------------------------------------
% The main remaining issue appears to lie in the choice-likelihood pathway,
% rather than in the clean forward generator itself. In particular, the full
% inversion currently does not recover the latent offset `p.c` in a way that
% is consistent with how synthetic datasets were generated.
%
% Future work should therefore focus on:
%
%   1. auditing the semantic meaning of `c` in the full inversion relative
%      to the forward simulation code
%   2. checking whether the current choice-likelihood term is evaluating the
%      right quantity for Flanker
%   3. determining whether the response modality and preference structure
%      inherited from the Stroop logic are the correct recovery target for
%      the Flanker model
%   4. revisiting the RT likelihood only after the choice-only inversion is
%      behaving sensibly
%
% For these reasons, this section is retained as an experimental
% implementation and diagnostic tool, but not yet as a finalized recovery
% result.
% =========================================================

%% ---------------------------------------------------------
% Full inversion over synthetic Flanker datasets
% ---------------------------------------------------------
% This section performs a Stroop-inspired full inversion for the Flanker
% model, using the clean unsolved 6-block template `FLK_TEMPLATE`.
%
% For each synthetic dataset:
%   1. observed outcomes are passed block-by-block into Y.o
%   2. stacked RT data are stored in Y.r (currently not used because M.rt=0)
%   3. spm_nlsi_Newton inverts the model to estimate latent offsets c and e
%   4. posterior means and uncertainty are stored
%   5. recovery is visualized online across datasets
%
% Important:
%   - M.ch = 1 means the inversion currently uses choice data
%   - M.rt = 0 means RT is excluded for now because the RT pathway is not
%     yet validated in the Flanker full inversion
% ---------------------------------------------------------

clear M
U  = [];
Ep = [];
Cp = [];

% Inversion model:
%   M.L = Flanker likelihood function
%   M.G = Flanker generator acting on the clean unsolved schedule template
M.L       = @MDP_Flanker_L;
M.G       = @(p) MDP_Flanker_Gen(p, FLK_TEMPLATE);

% Priors over latent parameters to be recovered
% p.c and p.e are recovered as local latent offsets
M.pE.c    = 0;                         % prior mean for c
M.pE.e    = 0;                         % prior mean for e
M.pC      = eye(spm_length(M.pE))/256; % prior covariance (tight prior)

% Include choice data, exclude RT for now
M.ch      = 1;                         % include choice likelihood
M.rt      = 0;                         % exclude RT likelihood for now

% Loop over all synthetic datasets / grid points
for i = 1:size(MDP,1)

    % -----------------------------------------------------
    % Build observed data structure for dataset i
    % -----------------------------------------------------
    % Each dataset consists of a full 6-block Flanker schedule.
    % Observed outcomes are passed blockwise as full 6 x T outcome matrices.
    Y.o = cell(1, size(MDP,2));
    for b = 1:size(MDP,2)
        Y.o{b} = MDP(i,b).o;
    end

    % Keep stacked RT vector for compatibility with the inversion
    % structure, even though M.rt = 0 currently disables its use.
    Y.r = R(:,i);

    % -----------------------------------------------------
    % Invert dataset i
    % -----------------------------------------------------
    [EP,CP,~] = spm_nlsi_Newton(M,U,Y);

    display(['Inverted model ' num2str(i) '/' num2str(size(MDP,1))])

    % Store posterior means and covariance summaries
    % Ep(i,:) = [c_hat, e_hat]
    % Cp(i,:) = [Var(c), Var(e), Cov(c,e)]
    Ep(i,:) = spm_vec(EP)';
    Cp(i,:) = [diag(CP)' CP(1,2)];

    % -----------------------------------------------------
    % Online recovery plots
    % -----------------------------------------------------
    % Plot current recovered parameters against true generating values
    % as inversion proceeds across datasets.
    spm_figure('GetWin','Figure 5'); clf

    % c recovery
    subplot(3,1,1)
    bar(1:size(Ep,1),Ep(:,1),'FaceColor',[.7 .7 .9],'EdgeColor',[1 1 1]), hold on
    errorbar(1:size(Ep,1),Ep(:,1),1.65*sqrt(Cp(:,1)),-1.65*sqrt(Cp(:,1)), ...
        'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
    plot(1:size(Ep,1),X(1:i,1),'.r','MarkerSize',20)
    hold off
    xlabel('Dataset')
    ylabel('Parameter estimate')
    title('c')

    % e recovery
    subplot(3,1,2)
    bar(1:size(Ep,1),Ep(:,2),'FaceColor',[.7 .7 .9],'EdgeColor',[1 1 1]), hold on
    errorbar(1:size(Ep,1),Ep(:,2),1.65*sqrt(Cp(:,2)),-1.65*sqrt(Cp(:,2)), ...
        'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
    plot(1:size(Ep,1),X(1:i,2),'.r','MarkerSize',20)
    hold off
    xlabel('Dataset')
    ylabel('Parameter estimate')
    title('e')

    % tradeoff recovery under the convention c - e
    subplot(3,1,3)
    bar(1:size(Ep,1),Ep(:,1)-Ep(:,2),'FaceColor',[.7 .7 .9],'EdgeColor',[1 1 1])
    hold on
    errorbar(1:size(Ep,1),Ep(:,1)-Ep(:,2), ...
        1.65*sqrt(Cp(:,1)+Cp(:,2)-2*Cp(:,3)), ...
       -1.65*sqrt(Cp(:,1)+Cp(:,2)-2*Cp(:,3)), ...
        'CapSize',0,'LineWidth',2,'LineStyle','None','Color',[.3 .3 .6])
    plot(1:size(Ep,1),X(1:i,1)-X(1:i,2),'.r','MarkerSize',20)
    xlabel('Dataset')
    ylabel('Parameters')
    title('c - e')
    hold off
end

% ---------------------------------------------------------
% Save posterior estimates
% ---------------------------------------------------------
Z.Ep = Ep;
Z.Cp = Cp;
save('Z','Z')