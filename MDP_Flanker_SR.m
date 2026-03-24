function [stim, resp] = MDP_Flanker_SR(MDP)
% This routine reports stimulus and response sequences without generating an
% animation.
%
% Flanker analogue of MDP_Stroop_SR.
%
% Output:
%   stim.Fdir : flanker-direction labels
%   stim.Tdir : target-direction labels
%   resp      : response labels

dirlab = {'left','right',' '};
rhlab  = {'left','right',' '};

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

        % In the current task structure, t=2 is the effective stimulus /
        % response time point, analogous to the Stroop helper.
        %
        % Outcome coding assumed:
        %   o(1,t) = flanker direction
        %   o(2,t) = target direction
        %   o(3,t) = cue
        %   o(4,t) = keypress
        %
        % Only keep non-null stimulus observations.
        if t == 2
            stim.Fdir = [stim.Fdir, {dirlab{MDP.o(1,t)}}];
            stim.Tdir = [stim.Tdir, {dirlab{MDP.o(2,t)}}];
            resp      = [resp, {['"' rhlab{MDP.o(4,t)} '"']}];
        end
    end
end