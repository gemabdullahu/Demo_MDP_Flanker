%% ---------------------------------------------------------
% Stroop-aligned pooled surrogate analysis
% ---------------------------------------------------------
% This section implements the pooled-summary surrogate workflow in the same
% spirit as `DEMO_MDP_Stroop`.
%
% For each parameter setting in the local Flanker grid, the script first
% simulates behavior across the full 6-block schedule and then constructs:
%
%   B(k)   = one pooled scalar accuracy summary
%   R(:,k) = one stacked RT-proxy vector across the full schedule
%
% The pooled RT summary used for surrogate fitting is the columnwise mean of
% R, exactly analogous in structure to the Stroop code. In that sense, this
% is the Flanker analogue of the Stroop pooled-summary stage:
%
%   latent parameters (c,e)  ->  pooled accuracy / pooled RT
%
% The purpose of this part is not yet to exploit the full context structure
% of the Flanker task, but rather to preserve methodological continuity with
% the Stroop surrogate framework. It therefore asks the same broad question:
%
%   can a low-order polynomial surrogate capture the mapping from latent
%   parameter variation to pooled behavioral summaries?
%
% In addition to the forward surrogate fit, this section also includes an
% explicit inverse-recovery step from the pooled scalar accuracy surrogate.
% This recovery step is not explicitly printed in the original Stroop demo,
% but it is a natural extension of the same surrogate framework and is used
% here to assess how well the pooled Flanker summaries recover the latent
% coordinates.
%
% THE PARAMETER GRID
% --------------------------
% Unlike the original Stroop demo, the Flanker grid here is defined as a
% local perturbation around calibrated baseline parameter values rather than
% a broad sweep over the full absolute parameter space.
%
% Specifically:
%   c0 = 0.22
%   e0 = 0.051366
%
% and the grid X is applied in log-space:
%   p.c = log(c0) + X(i,1)
%   p.e = log(e0) + X(i,2)
%
% This means the surrogate is built over a local neighborhood of a working
% operating point. This was done because the Flanker implementation proved
% more stable and behaviorally interpretable when explored around a
% calibrated baseline regime rather than over a wider unconstrained range.
%
% As a consequence, the current grid should be interpreted as a local
% sensitivity / recovery grid, not as an exhaustive global parameter sweep.
% It also helps explain why some summaries (especially congruent conditions)
% remain relatively flat over the explored range.
%
% RT proxy limitation
% --------------------------
% The pooled RT matrix R was constructed in a manner structurally homologous
% to the Stroop implementation: one stacked RT vector per grid point, with
% the pooled RT summary defined as the columnwise mean. However, in the present
% Flanker implementation the RT proxy showed only weak and irregular variation
% across the parameter grid. As a result, the true pooled RT surface had a much
% smaller dynamic range than the accuracy surface and did not exhibit a clear
% smooth gradient over the latent coordinates. When the Stroop-style low-order
% polynomial surrogate was fit to these pooled RT summaries, it collapsed toward
% an almost flat surface, often appearing as a single color in the visualization.
% In other words, the RT stage is structurally Stroop-aligned but functionally weak
% in the current Flanker model. We therefore need help specifically with the RT-proxy
% formulation and/or response-level model dynamics, rather than with the pooled
% summary construction. The main open issue is how to obtain an RT-like quantity
% that varies smoothly and meaningfully across the Flanker parameter grid, in a way
% that preserves Stroop-style methodological alignment while improving informativeness
% for surrogate fitting and recovery.

clearvars -except OPTIONS
clc;

if ~exist('OPTIONS','var')
    OPTIONS = struct('gamma',1,'plot',0);
end

rng default

%% ---------------------------------------------------------
% DESIGN MATRIX AND GRID-BASED MODEL SIMULATION
% ---------------------------------------------------------
% The grid X defines the surrogate design space used for simulation and
% recovery. Unlike a broad unconstrained parameter sweep, this grid is
% defined as local log-space perturbations around calibrated baseline
% values c0 and e0.
%
% In other words:
%   p.c = log(c0) + X(i,1)
%   p.e = log(e0) + X(i,2)
%
% so X should be interpreted as a local sensitivity / recovery grid around
% a working operating point rather than the full absolute parameter space.
%
% Columns of X:
%   X(:,1) = c-axis offsets
%   X(:,2) = e-axis offsets
%
% The present grid matches the Stroop-style polynomial-surrogate logic:
% a small 2D design is used to generate synthetic summaries over nearby
% parameter settings, after which a surrogate approximation is fitted.
% ----------------------------------------------------------
[x, y] = meshgrid((-1:1/2:1)/4, (-1:1/2:1)/4);
X = [x(:), y(:)];   % col 1 = local c offset, col 2 = local e offset

% Calibrated baseline values around which the local grid is constructed.
% These were chosen as a plausible working operating point for the current
% Flanker implementation.
c0 = 0.22;
e0 = 0.051366;

% Block schedule used for simulation.
% Context coding:
%   1 = MC (mostly congruent)
%   2 = MI (mostly incongruent)
%   3 = NP (balanced / non-predictive)
%
% The 6-block schedule mirrors the block-based task structure used in the
% Flanker simulations:
%   MC -> MI -> NP -> MI -> MC -> NP
ctx_blocks = [1 2 3 2 1 3];
T_blocks   = [60 60 30 60 60 30];
ctx_names  = {'MC','MI','NP'};

% Loop over all parameter settings in the design grid.
% For each grid point:
%   1. instantiate the model with local c/e perturbations
%   2. generate a template MDP
%   3. solve the model separately for each block in the schedule
for i = 1:size(X,1)

    p = struct();

    % Local parameterization:
    % c and e are varied in log-space around the baseline operating point.
    % This keeps the explored regime close to the calibrated behaviorally
    % plausible region of the model.
    p.c   = log(c0) + X(i,1);
    p.e   = log(e0) + X(i,2);

    % Context-specific congruency probabilities:
    %   MC = 0.75 congruent
    %   MI = 0.25 congruent
    %   NP = 0.50 congruent
    p.pc  = [0.75 0.25 0.50];

    % Return a template model only; the actual solving is done below with
    % spm_MDP_VB_X for each block separately.
    p.run = false;

    % Solver / model settings.
    p.tau = 8;

    % Explicit motor-noise parameter.
    % eta softens the final response mapping; here it is set to a small
    % value to preserve informative variation while avoiding an overly rigid
    % response channel.
    p.eta = 0.1;

    fprintf('\n--- parameter setting %d/%d ---\n', i, size(X,1));
    fprintf('p.c = %.4f | p.e = %.4f\n', p.c, p.e);

    % Build the template deep MDP for the current parameter setting.
    mdp0 = FLK_Online_3(p);

    % Reset the random stream so that blockwise stochasticity is comparable
    % across parameter settings.
    rng default

    % Solve each block in the schedule under the current parameter setting.
    for b = 1:numel(ctx_blocks)

        mdp = mdp0;
        mdp.T = T_blocks(b);

        % Set the contextual state at the slow level.
        % The second D factor codes context:
        %   [1;0;0] = MC
        %   [0;1;0] = MI
        %   [0;0;1] = NP
        if ctx_blocks(b) == 1
            mdp.D{2} = [1;0;0];   % MC
        elseif ctx_blocks(b) == 2
            mdp.D{2} = [0;1;0];   % MI
        else
            mdp.D{2} = [0;0;1];   % NP
        end

        % Solve the model for this parameter setting and block.
        MDP(i,b) = spm_MDP_VB_X(mdp, OPTIONS);
    end
end

fprintf('\nGeneration complete.\n');
fprintf('size(MDP) = [%d %d]\n', size(MDP,1), size(MDP,2));


%% ---------------------------------------------------------
% PART 1. Stroop-aligned pooled performance summaries
% ---------------------------------------------------------
% This section implements the pooled-summary stage in the same spirit as
% DEMO_MDP_Stroop.
%
% For each parameter setting k, we construct:
%
%   B(k)     = one pooled scalar accuracy summary
%   R(:,k)   = one stacked RT-proxy vector across the full schedule
%
% This is the Flanker analogue of the Stroop pooled-summary logic:
% rather than keeping condition-specific summaries at this stage, we first
% collapse behavior into one scalar accuracy and one pooled RT summary per
% grid point before fitting the polynomial surrogate.
%
% In Stroop, RTs are stacked across the two task conditions. Here, RTs are
% stacked across the full 6-block Flanker schedule.
% ---------------------------------------------------------

% Total number of usable deep trials across the full simulated schedule.
% Each block contributes (T - 1) effective trials because the first time
% point corresponds to initialization rather than a behavioral trial.
nRT = 0;
for b = 1:size(MDP,2)
    nRT = nRT + (MDP(1,b).T - 1);
end

% Preallocate pooled summaries:
%
% B : one pooled scalar accuracy value per parameter setting
% R : stacked RT-proxy matrix
%     - rows    = all effective trials across the full schedule
%     - columns = parameter settings / grid points
B = zeros(1, size(X,1));
R = nan(nRT, size(X,1));

% Loop over all parameter settings in the design grid.
for k = 1:size(MDP,1)

    % acc_sum accumulates blockwise accuracy contributions for the pooled
    % scalar accuracy summary B(k).
    %
    % rt_idx0 tracks the write position in the stacked RT vector R(:,k).
    acc_sum = 0;
    rt_idx0 = 1;

    % Loop over all blocks in the full Flanker schedule.
    for b = 1:size(MDP,2)

        % Extract trialwise target/flanker directions and responses in a
        % compact behavioral format.
        [stim{k,b}, resp{k,b}] = MDP_Flanker_SR(MDP(k,b));

        % H rows:
        %   row 1 = congruent?   (1 = yes, 0 = no)
        %   row 2 = correct?     (1 = yes, 0 = no)
        H = zeros(2, numel(resp{k,b}));
        for i = 1:numel(resp{k,b})
            H(1,i) = strcmp(stim{k,b}.Tdir{i}, stim{k,b}.Fdir{i});      % congruent?
            H(2,i) = strcmp(['"' stim{k,b}.Tdir{i} '"'], resp{k,b}{i}); % correct?
        end

        % Add mean block accuracy to the pooled scalar accuracy summary.
        % After the full schedule, B(k) becomes the average block accuracy
        % across all simulated blocks for the current parameter setting.
        acc_sum = acc_sum + sum(H(2,:)) / length(H(2,:));

        % Compute RT-like proxy values for the current block.
        %
        % As in the Stroop implementation, RTs are transformed with an
        % exponential mapping and a small Gaussian perturbation so that they
        % behave more like noisy behavioral response times rather than
        % deterministic solver outputs.
        rt = MDP_Flanker_RT(MDP(k,b));
        rt = exp(rt(:) + randn(size(rt(:)))/16) / 2;

        % Append the current block's RTs into the stacked RT matrix.
        nThis = numel(rt);
        R(rt_idx0:rt_idx0+nThis-1, k) = rt;
        rt_idx0 = rt_idx0 + nThis;
    end

    % Final pooled scalar accuracy for parameter setting k:
    % average of blockwise accuracies across the full schedule.
    B(k) = acc_sum / size(MDP,2);
end


%% ---------------------------------------------------------
% Save generated summaries for later recovery runs
% ---------------------------------------------------------
save('flk_generated_summaries.mat', ...
    'X', 'MDP', 'B', 'R');

fprintf('\nSaved generated summaries to flk_generated_summaries.mat\n');

%% ---------------------------------------------------------
% Stroop-aligned pooled polynomial surrogate fit
% ---------------------------------------------------------
% This section fits low-order polynomial surrogate models to the pooled
% behavioral summaries constructed above.
%
% As in DEMO_MDP_Stroop, the surrogate is defined over a 2D latent design
% matrix U built from:
%
%   - intercept
%   - linear c term
%   - linear e term
%   - c-by-e interaction
%   - quadratic c term
%   - quadratic e term
%
% The surrogate is fit separately for:
%
%   1. pooled scalar accuracy summary B
%   2. pooled scalar RT summary mean(R)
%
% The goal is to approximate the forward mapping:
%
%   latent parameters (c,e)  ->  pooled behavioral summaries
%
% before any later recovery analysis.
% ---------------------------------------------------------

% Polynomial design matrix over the latent parameter grid.
% Columns:
%   1 = intercept
%   2 = linear c term
%   3 = linear e term
%   4 = c*e interaction
%   5 = c^2
%   6 = e^2
U = [ones(size(X,1),1), X, X(:,1).*X(:,2), X(:,1).^2, X(:,2).^2];

% ---------------------------------------------------------
% Accuracy surrogate
% ---------------------------------------------------------
% Fit a polynomial surrogate to the pooled scalar accuracy summary B.
%
% The forward model uses a logistic transform so that predicted values stay
% in the probability range.
%
% Inputs:
%   U      = polynomial design matrix over grid points
%   P.beta = polynomial coefficients
%   P.pi   = log precision parameter controlling residual variance
%
% Y is passed as B' so that it matches the column-vector shape of the
% surrogate predictions, exactly as in the Stroop code style.
M.L       = @(P,M,U,Y) sum(log( spm_Npdf(Y, exp(U*P.beta)./(exp(U*P.beta)+exp(1)), exp(-P.pi)) ));

% Priors on surrogate parameters.
% beta = polynomial coefficients
% pi   = log precision of observation noise
M.pE.beta = zeros(size(U,2),1);                   % prior mean of coefficients
M.pE.pi   = 4;                                    % prior log precision
M.pC      = eye(spm_length(M.pE));                % prior covariance

% Variational fit of the surrogate to pooled accuracy.
[Ep,Cp,F] = spm_nlsi_Newton(M,U,B');

% Bayesian model reduction / averaging over polynomial terms.
% This provides posterior inclusion probabilities for the surrogate terms.
DCM.Ep = Ep;
DCM.Cp = Cp;
DCM.F  = F;
DCM.M  = M;
[PCM,~,BMA] = spm_dcm_bmr_all(DCM,'beta');

% Accuracy-surrogate outputs:
%   Bp  = posterior mean polynomial coefficients
%   BPp = posterior probabilities for polynomial terms
Bp  = BMA.Ep.beta;
BPp = PCM.Pp.beta;

% ---------------------------------------------------------
% RT surrogate
% ---------------------------------------------------------
% Fit a polynomial surrogate to the pooled scalar RT summary.
%
% As in the Stroop implementation, the surrogate is fit to log RT rather
% than raw RT. The pooled RT summary is obtained as the columnwise mean of
% the stacked RT matrix R.
%
% mean(R)' is the direct Stroop-style pooled RT input, where each grid
% point contributes one scalar RT summary.
M.L       = @(P,M,U,Y) sum(log( spm_Npdf(log(Y), U*P.beta, exp(-P.pi)) ));

% Variational fit of the surrogate to pooled mean RT.
[Ep,Cp,F] = spm_nlsi_Newton(M,U,mean(R)');

% Bayesian model reduction / averaging for the RT surrogate.
DCM.Ep = Ep;
DCM.Cp = Cp;
DCM.F  = F;
DCM.M  = M;
[PCM,~,BMA] = spm_dcm_bmr_all(DCM,'beta');

% RT-surrogate outputs:
%   Rp  = posterior mean polynomial coefficients
%   RPp = posterior probabilities for polynomial terms
Rp  = BMA.Ep.beta;
RPp = PCM.Pp.beta;

%% ---------------------------------------------------------
% Parameter recovery from Stroop-aligned pooled polynomial surrogate
% ---------------------------------------------------------
% This section performs an explicit inverse-recovery analysis using the
% pooled scalar accuracy surrogate fitted above.
%
% Important note:
% this recovery step is not explicitly implemented in DEMO_MDP_Stroop
% itself. Rather, it is a natural extension of the same pooled surrogate
% framework. The idea is:
%
%   1. simulate pooled scalar accuracy summaries B over the parameter grid
%   2. fit the Stroop-style polynomial surrogate to B
%   3. invert that fitted surrogate for each synthetic summary B(k)
%   4. compare recovered parameters to the true grid values
%
% Thus, this section evaluates how well the pooled scalar surrogate can
% recover the latent coordinates c and e from the synthetic summaries.
% ---------------------------------------------------------
nGrid = size(X,1);

% True latent coordinates used to generate the grid.
% By construction:
%   X(:,1) = c-axis offsets
%   X(:,2) = e-axis offsets
c_true = X(:,1);
e_true = X(:,2);

% Parr-style tradeoff convention used here:
%   tradeoff = c - e
d_true = c_true - e_true;

% Preallocate recovered parameter estimates from the pooled scalar
% surrogate inversion.
c_hat_scalar = nan(nGrid,1);
e_hat_scalar = nan(nGrid,1);

% Recover parameters separately for each synthetic grid point.
for k = 1:nGrid

    % Observed pooled scalar accuracy summary for this grid point.
    y_obs = B(k);

    % Define a small inversion model for scalar recovery.
    % The fitted surrogate coefficients Bp are treated as fixed, and the
    % unknowns are the latent coordinates c and e that best reproduce y_obs.
    Msub = struct();
    Msub.Bp = Bp;
    Msub.pE.c = 0;
    Msub.pE.e = 0;
    Msub.pC   = eye(2)/4;
    Msub.L    = @L_poly_scalar;

    % Variational inversion of the scalar surrogate at this grid point.
    [EP, ~, ~] = spm_nlsi_Newton(Msub, [], y_obs);

    % Store recovered latent coordinates.
    c_hat_scalar(k) = EP.c;
    e_hat_scalar(k) = EP.e;
end

% Recovered tradeoff under the same convention used for d_true.
d_hat_scalar = c_hat_scalar - e_hat_scalar;

% Print per-grid true vs recovered values.
fprintf('\n===== RECOVERED PARAMETERS: STROOP-ALIGNED SCALAR POLYNOMIAL =====\n');
disp(table((1:nGrid)', c_true, e_true, c_hat_scalar, e_hat_scalar, d_true, d_hat_scalar, ...
    'VariableNames', {'GridIdx','c_true','e_true','c_hat','e_hat','c_minus_e_true','c_minus_e_hat'}));

% Print compact recovery diagnostics.
fprintf('\nRecovery summary:\n');
fprintf('corr(c_true, c_hat)   = %.3f\n', corr(c_true, c_hat_scalar, 'rows','complete'));
fprintf('corr(e_true, e_hat)   = %.3f\n', corr(e_true, e_hat_scalar, 'rows','complete'));
fprintf('corr((c-e), recovered)= %.3f\n', corr(d_true, d_hat_scalar, 'rows','complete'));

%% ---------------------------------------------------------
% Visualization of pooled summaries and fitted surrogate surfaces
% ---------------------------------------------------------
% These plots mirror the original Stroop-style presentation:
%
%   top-left  : true pooled accuracy surface
%   top-right : true pooled mean RT surface
%   bottom-left  : fitted pooled accuracy surrogate
%   bottom-right : fitted pooled RT surrogate
%   bottom row   : posterior probabilities for surrogate terms
%
% Together, these panels show:
%   - how the simulated pooled summaries vary across the grid
%   - how well the polynomial surrogate reproduces that variation
%   - which polynomial terms are supported by Bayesian model reduction
spm_figure('GetWin','Figure 4'); clf

subplot(3,2,1)
imagesc(X(:,1),X(:,2),100*reshape(B,[sqrt(length(B)),sqrt(length(B))]))
title('Percentage correct')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')

subplot(3,2,2)
imagesc(X(:,1),X(:,2),reshape(mean(R),[sqrt(length(mean(R))),sqrt(length(mean(R)))]))
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
imagesc(X(:,1),X(:,2),reshape(exp(U*Rp),[sqrt(size(R,2)),sqrt(size(R,2))]))
title('Mean reaction time (fit)')
colorbar
axis square, axis xy
xlabel('c')
ylabel('e')
caxis([min(mean(R)),max(mean(R))])

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


%% ---------------------------------------------------------
% Local Functions
% ----------------------------------------------------------
function L = L_poly_scalar(P,M,~,Y)

% Scalar pooled-accuracy likelihood for surrogate inversion.
%
% Given candidate latent coordinates (c,e), compute the surrogate-predicted
% pooled scalar accuracy using the fitted polynomial coefficients Bp.
mu = exp([1, P.c, P.e, P.c*P.e, P.c^2, P.e^2] * M.Bp);
mu = mu / (mu + exp(1));

% Gaussian observation model on the pooled scalar summary.
sig = 0.05;
z = (Y - mu) / sig;

L = -0.5*(z.^2) - log(sig*sqrt(2*pi));
end

%% ---------------------------------------------------------
% RT proxy diagnostics: why the RT fit looks flat / unsmooth
% ---------------------------------------------------------

% pooled RT summary per grid point (same object used in the surrogate fit)
Rbar = mean(R,1)';

fprintf('size(Rbar) = [%d %d]\n', size(Rbar,1), size(Rbar,2));
fprintf('Rbar range = [%.6f %.6f]\n', min(Rbar), max(Rbar));
fprintf('Rbar std   = %.6f\n', std(Rbar));
fprintf('Rbar CV    = %.6f\n', std(Rbar) / mean(Rbar));

% compare true RT variation to fitted RT variation
Rfit = exp(U*Rp);   % this matches your current RT fit code
Rfit_grid = reshape(Rfit, [ng ng]);

fprintf('\n===== TRUE vs FITTED RT SURFACE VARIATION =====\n');
fprintf('True RT surface:  range = [%.6f %.6f], std = %.6f\n', ...
    min(Rbar), max(Rbar), std(Rbar));
fprintf('Fitted RT surface: range = [%.6f %.6f], std = %.6f\n', ...
    min(Rfit), max(Rfit), std(Rfit));

Bcol = B(:);
Bgrid = reshape(Bcol, [ng ng]);

Dbc = diff(Bgrid,1,2);
Dbe = diff(Bgrid,1,1);

fprintf('\n===== ACCURACY vs RT COMPARISON =====\n');
fprintf('Accuracy range = %.6f | std = %.6f\n', max(Bcol)-min(Bcol), std(Bcol));
fprintf('RT range       = %.6f | std = %.6f\n', max(Rbar)-min(Rbar), std(Rbar));

fprintf('Accuracy mean abs diff (c,e) = %.6f , %.6f\n', ...
    mean(abs(Dbc(:))), mean(abs(Dbe(:))));
fprintf('RT mean abs diff (c,e)       = %.6f , %.6f\n', ...
    mean(abs(Dc(:))), mean(abs(De(:))));