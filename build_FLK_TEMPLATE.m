%% ---------------------------------------------------------
% Clean unsolved 6-block Flanker template for full inversion
% ---------------------------------------------------------
% The full inversion no longer uses an already solved MDP object as its
% template. Instead, it rebuilds a clean unsolved 6-block schedule so that
% candidate parameter changes are expressed cleanly during inversion.

p0 = struct();
p0.c   = log(0.22);
p0.e   = log(0.051366);
p0.pc  = [0.75 0.25 0.50];
p0.run = false;
p0.tau = 8;
p0.eta = 0.010;

mdp0 = FLK_Online_3(p0);

FLK_TEMPLATE = repmat(mdp0, 1, 6);

ctx_blocks = [1 2 3 2 1 3];
T_blocks   = [60 60 30 60 60 30];

for b = 1:6
    FLK_TEMPLATE(b) = mdp0;
    FLK_TEMPLATE(b).T = T_blocks(b);

    if ctx_blocks(b) == 1
        FLK_TEMPLATE(b).D{2} = [1;0;0];   % MC
    elseif ctx_blocks(b) == 2
        FLK_TEMPLATE(b).D{2} = [0;1;0];   % MI
    else
        FLK_TEMPLATE(b).D{2} = [0;0;1];   % NP
    end
end