function rt = MDP_Flanker_RT(MDP)
% Simulate RT-like values from predictive entropy
% Flanker analogue of MDP_Stroop_RT.
%
% Assumes:
%   - MDP.mdp(i).xn contains posterior beliefs over hidden states
%   - A{4} is the response / keypress outcome modality
%
% Output:
%   rt(i-1) = final predictive entropy of the response outcome for trial i

rt = nan(1, length(MDP.mdp)-1);

for i = 2:length(MDP.mdp)

    x = [];

    % loop over VB updates / posterior samples within trial i
    for k = 1:size(MDP.mdp(i).xn{1},1)

        xn = cell(1, numel(MDP.mdp(i).xn));

        % extract posterior beliefs for each hidden-state factor
        for j = 1:numel(MDP.mdp(i).xn)
            xn{j} = MDP.mdp(i).xn{j}(k,:,2,1);
        end

        % predicted response outcome distribution
        x(:,end+1) = spm_dot(MDP.mdp(i).A{4}, xn);
    end

    % predictive entropy over response outcomes at each update
    v = -diag(x' * spm_log(x));

    % Stroop code keeps the final entropy as RT proxy
    rt(i-1) = v(end);
end