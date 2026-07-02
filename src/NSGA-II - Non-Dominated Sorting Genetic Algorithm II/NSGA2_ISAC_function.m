function result = NSGA2_ISAC_function(params, seed)
    rng(seed);

    % ============================================================
    % 1) GA PARAMETERS
    % ============================================================
    Npop    = 50;
    MaxIter = 500;
    eta     = params.eta;

    pc = 0.85;      % crossover probability
    pm = 0.15;      % mutation probability
    tournamentK = 3;

    mut_amp0   = 0.06;   % base mutation amplitude
    mut_ampMin = 0.008;  % late-stage mutation amplitude

    % ============================================================
    % 2) INIT POPULATION
    % ============================================================
    Pop = repmat(struct( ...
        'X',        [], ...
        'Obj',      [], ...
        'Rank',     [], ...
        'Distance', []), Npop, 1);

    Hdl = params.h_dl;      % size: Nt x M
    a_t = params.a_t(:);

    Nt = params.Nt;
    M  = params.M;

    eps_reg = 1e-9;

    % ==========================================================
    % 1) ZF communication precoder
    % Vc_tilde = Hdl (Hdl^H Hdl)^-1
    % ==========================================================
    Vc_tilde = Hdl / (Hdl' * Hdl + eps_reg*eye(M));

    % Unit Frobenius normalization
    Vc_bar = Vc_tilde / norm(Vc_tilde,'fro');

    Vc0 = sqrt(eta * params.P_DL) * Vc_bar;

    % ==========================================================
    % 2) Null-space sensing waveform
    % P_perp = I - Hdl(Hdl^H Hdl)^-1 Hdl^H
    % ==========================================================
    P_perp = eye(Nt) - Hdl / (Hdl' * Hdl + eps_reg*eye(M)) * Hdl';

    % Projection of steering vector
    vs_tilde = P_perp * a_t;
    nv = norm(vs_tilde);

    if nv < 1e-12
        vs_bar = a_t / norm(a_t);
    else
        vs_bar = vs_tilde / nv;
    end

    % Rank-1 sensing matrix
    Vs_bar = zeros(Nt, Nt);
    Vs_bar(:,1) = vs_bar;

    % Initial sensing power split
    Vs0 = sqrt((1-eta) * params.P_DL) * Vs_bar;

    for n = 1:Npop
        if n < 2
            X.Vc  = Vc0;
            X.Vs  = Vs0;

            % Equal UL power allocation
            X.rho = params.P_UL * ones(params.Pnum,1);

        elseif n <= 20
            % Slightly perturbed version of n = 1
            noise_level = 0.005;

            Vc_rand = randn(params.Nt, params.M) ...
                    + 1j*randn(params.Nt, params.M);

            Vs_rand = randn(params.Nt, params.Nt) ...
                    + 1j*randn(params.Nt, params.Nt);

            rho_rand = randn(params.Pnum,1);

            % Add small perturbation
            X.Vc = Vc0 + noise_level * Vc_rand;
            X.Vs = Vs0 + noise_level * Vs_rand;

            X.rho = params.P_UL * ones(params.Pnum,1) ...
                  + noise_level * abs(rho_rand);

            % Re-normalize powers
            X.Vc = X.Vc / norm(X.Vc,'fro') * sqrt(eta * params.P_DL);
            X.Vs = X.Vs / norm(X.Vs,'fro') * sqrt((1-eta) * params.P_DL);

            % Clip UL powers
            X.rho = max(X.rho, 0);
            X.rho = X.rho / sum(X.rho) * params.P_UL;

        else
            % Random initialization
            X.Vc = randn(params.Nt, params.M) + 1j*randn(params.Nt, params.M);
            X.Vs = randn(params.Nt, params.Nt) + 1j*randn(params.Nt, params.Nt);
            X.rho = params.P_UL * rand(params.Pnum,1);

            X.Vc = X.Vc / norm(X.Vc,'fro') * sqrt(eta * params.P_DL);
            X.Vs = X.Vs / norm(X.Vs,'fro') * sqrt((1-eta) * params.P_DL);
        end

        % Clamp rho
        X.rho = min(max(X.rho,0),params.P_UL);

        % Refresh
        X = refresh_solution_state(X, params);

        % Evaluate objectives
        Obj = evaluateObjectives(X, params);

        % Store
        Pop(n).X        = X;
        Pop(n).Obj      = Obj;
        Pop(n).Rank     = inf;
        Pop(n).Distance = 0;
    end

    % Assign Pareto rank & crowding distance
    Pop = assignRankAndDistance(Pop);

    % ============================================================
    % 3) HISTORY
    % ============================================================
%     B_history      = zeros(MaxIter,1);
%     RadarSINR_hist = zeros(MaxIter,1);
%     ULSINR_hist    = zeros(MaxIter, params.Pnum);
%     DLSINR_hist    = zeros(MaxIter, params.M);
    Obj_history = cell(MaxIter,1);

    % ============================================================
    % 4) MAIN LOOP
    % ============================================================
    for it = 1:MaxIter

        %--------------------------------------------
        % Mutation amplitude
        %--------------------------------------------
        mut_amp = mut_amp0 * (1 - it/MaxIter) ...
                + mut_ampMin * (it/MaxIter);
    
        %--------------------------------------------
        % Generate offspring population
        %--------------------------------------------
        NewPop = repmat(Pop(1), Npop, 1);
    
        k = 1;
    
        while k <= Npop
    
            p1 = tournament_select(Pop, tournamentK);
            p2 = tournament_select(Pop, tournamentK);
    
            X1 = Pop(p1).X;
            X2 = Pop(p2).X;
    
            if rand < pc
                [C1,C2] = crossover_isac(X1,X2,params);
            else
                C1 = X1;
                C2 = X2;
            end
    
            C1 = mutate_isac(C1,params,mut_amp,pm);
            C2 = mutate_isac(C2,params,mut_amp,pm);
    
            C1 = refresh_solution_state(C1,params);
            C2 = refresh_solution_state(C2,params);
    
            Obj1 = evaluateObjectives(C1,params);
            Obj2 = evaluateObjectives(C2,params);
    
            NewPop(k).X = C1;
            NewPop(k).Obj = Obj1;
            NewPop(k).Rank = inf;
            NewPop(k).Distance = 0;
    
            k = k+1;
    
            if k<=Npop
    
                NewPop(k).X = C2;
                NewPop(k).Obj = Obj2;
                NewPop(k).Rank = inf;
                NewPop(k).Distance = 0;
    
                k = k+1;
    
            end
    
        end
    
        %--------------------------------------------
        % Merge parent + offspring
        %--------------------------------------------
        MergePop = [Pop;NewPop];
    
        %--------------------------------------------
        % Fast Non-dominated Sorting
        %--------------------------------------------
        MergePop = assignRankAndDistance(MergePop);
    
        %--------------------------------------------
        % Environmental Selection
        %--------------------------------------------
        Pop = environmentalSelection(MergePop,Npop);
    
        %--------------------------------------------
        % Save history
        %--------------------------------------------
        Obj_history{it} = vertcat(Pop.Obj);
    
    end

    % ============================================================
    % 5) RETURN
    % ============================================================
%     result.Best           = Best;
%     result.B_history      = B_history;
%     result.RadarSINR_hist = RadarSINR_hist;
%     result.ULSINR_hist     = ULSINR_hist;
%     result.DLSINR_hist     = DLSINR_hist;
%     result.params         = params;
    result.Pop         = Pop;
    result.Obj_history = Obj_history;
    result.params      = params;
end

% ============================================================
% CROSSOVER
% ============================================================
function [C1, C2] = crossover_isac(X1, X2, params)
    alpha = rand();

    C1.Vc  = alpha * X1.Vc  + (1-alpha) * X2.Vc;
    C2.Vc  = alpha * X2.Vc  + (1-alpha) * X1.Vc;

    C1.Vs  = alpha * X1.Vs  + (1-alpha) * X2.Vs;
    C2.Vs  = alpha * X2.Vs  + (1-alpha) * X1.Vs;

    C1.rho = alpha * X1.rho + (1-alpha) * X2.rho;
    C2.rho = alpha * X2.rho + (1-alpha) * X1.rho;

    % clamp rho
    C1.rho = min(max(C1.rho,0),params.P_UL);
    C2.rho = min(max(C2.rho,0),params.P_UL);
end

% ============================================================
% MUTATION
% ============================================================
function X = mutate_isac(X, params, mut_amp, pm)
    if rand < pm
        X.Vc = X.Vc + mut_amp * (randn(size(X.Vc)) + 1j*randn(size(X.Vc)));
    end

    if rand < pm
        X.Vs = X.Vs + mut_amp * (randn(size(X.Vs)) + 1j*randn(size(X.Vs)));
    end

    if rand < pm
        X.rho = X.rho + mut_amp * randn(size(X.rho));
        X.rho = min(max(X.rho,0),params.P_UL);
    end
end

% ============================================================
% REPAIR / REFRESH STATE
% ============================================================
function X = refresh_solution_state(X, params)

    eps0 = 1e-12;

    % =========================================================
    % 0) INIT U and w if missing
    % =========================================================
    if ~isfield(X,'U') || isempty(X.U)
        X.U = cell(params.Pnum,1);
        for pp = 1:params.Pnum
            hp = params.H_ul(:,pp);
            X.U{pp} = hp / (norm(hp) + eps0);
        end
    end

    if ~isfield(X,'w') || isempty(X.w)
        X.w = params.a_r / (norm(params.a_r) + eps0);
    end

    % =========================================================
    % 1) TX covariance + power constraint
    % =========================================================
    Rtx = X.Vc*X.Vc' + X.Vs*X.Vs';

    P_used = real(trace(Rtx));
    if P_used > params.P_DL
        scale = sqrt(params.P_DL / (P_used + eps0));
        X.Vc = X.Vc * scale;
        X.Vs = X.Vs * scale;
    end

    % recompute after scaling
    Rtx = X.Vc*X.Vc' + X.Vs*X.Vs';

    % =========================================================
    % 2) UPDATE UL RECEIVERS u_p  (MVDR)
    % =========================================================
    for pp = 1:params.Pnum

        Gp = zeros(params.Nr, params.Nr);

        % UL multi-user interference
        for i = 1:params.Pnum
            if i ~= pp
                hi = params.H_ul(:,i);
                Gp = Gp + X.rho(i) * (hi * hi');
            end
        end

        % TARGET
        A_t = params.A_target;
        Gp = Gp + abs(params.alpha_t)^2 * (A_t * Rtx * A_t');

        % CLUTTER
        for k = 1:length(params.theta_clutter)
            A_c = params.A_clutter(:,:,k);
            Gp = Gp + abs(params.alpha_clutter(k))^2 * (A_c * Rtx * A_c');
        end

        % NOISE
        Gp = Gp + params.sigma_u2 * eye(params.Nr);

        % MVDR update
        hp   = params.H_ul(:,pp);
        temp = Gp \ hp;
        den  = hp' * temp;

        X.U{pp} = temp / (den + eps0);
    end

    % =========================================================
    % 3) UPDATE RADAR RECEIVER w (MVDR)
    % =========================================================
    a_r = params.a_r;

    Rrad = params.sigma2 * eye(params.Nr);

    % CLUTTER
    for k = 1:length(params.theta_clutter)
        A_c = params.A_clutter(:,:,k);
        Rrad = Rrad + abs(params.alpha_clutter(k))^2 * (A_c * Rtx * A_c');
    end

    % UL interference
    for pp = 1:params.Pnum
        hp = params.H_ul(:,pp);
        Rrad = Rrad + X.rho(pp) * (hp * hp');
    end

    % MVDR
    temp = Rrad \ a_r;
    X.w = temp / (a_r' * temp + eps0);

end