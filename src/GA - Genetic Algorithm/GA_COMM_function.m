function result = GA_COMM_function(params, seed)
    rng(seed);

    % ============================================================
    % 1) GA PARAMETERS
    % ============================================================
    Npop    = 50;
    MaxIter = 1000;
    eta = params.eta;

    pc = 0.85;      % crossover probability
    pm = 0.15;      % mutation probability
    eliteCount = 2; % elitism
    tournamentK = 3;

    mut_amp0 = 0.08;   % base mutation amplitude
    mut_ampMin = 0.003; % late-stage mutation amplitude

    % ============================================================
    % 2) INIT POPULATION
    % ============================================================
    Pop = repmat(struct('X',[],'B',[],'Bbest',[],'Pbest',[]), Npop, 1);

    Hdl = params.h_dl;      % size: Nt x M
    
    Nt = params.Nt;
    M  = params.M;
    
    eps_reg = 1e-9;
    
    % ==========================================================
    % ZF communication precoder
    % Vc_tilde = Hdl (Hdl^H Hdl)^-1
    % ==========================================================
    
    Vc_tilde = Hdl / (Hdl' * Hdl + eps_reg*eye(M));
    
    % Unit Frobenius normalization
    Vc_bar = Vc_tilde / norm(Vc_tilde,'fro');
    
    % No sensing waveform in Comm-only
    Vc0 = sqrt(eta * params.P_DL) * Vc_bar;
    
    % ==========================================================
    % 2) Sensing
    % ==========================================================
    Vs0 = zeros(Nt, Nt);

    for n = 1:Npop
        if n < 8
            X.Vc  = Vc0;
            X.Vs  = Vs0;
            
            % Equal UL power allocation
            X.rho = params.P_UL * ones(params.Pnum,1);
        elseif n <= 24
            
            % ==========================================================
            % Slightly perturbed version of n = 1
            % ==========================================================
    
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
            X.Vc = X.Vc / norm(X.Vc,'fro') ...
                  * sqrt(eta * params.P_DL);
    
            X.Vs = X.Vs / norm(X.Vs,'fro') ...
                 * sqrt((1-eta) * params.P_DL);
    
            % Clip UL powers
            X.rho = max(X.rho, 0);
    
            X.rho = X.rho / sum(X.rho) * params.P_UL;
        else
            % ===== Random initialization =====
            X.Vc = randn(params.Nt, params.M) + 1j*randn(params.Nt, params.M);
            X.Vs = zeros(params.Nt, params.Nt);
            
            X.rho = params.P_UL * rand(params.Pnum,1);
    
            X.Vc  = X.Vc / norm(X.Vc,'fro') * sqrt(eta*params.P_DL);
            X.Vs  = X.Vs;
        end
    
        % ===== Clamp rho =====
        X.rho = min(max(X.rho,0),params.P_UL);
    
        % ===== Refresh =====
        X = refresh_solution_state(X, params);
    
        % ===== Store =====
        Pop(n).X     = X;
        Pop(n).B     = brightness(X, params);
        Pop(n).Pbest = X;
        Pop(n).Bbest = Pop(n).B;
    end

    [~,id] = max([Pop.B]);
    Best = Pop(id);

    % ============================================================
    % 3) HISTORY
    % ============================================================
    B_history      = zeros(MaxIter,1);
    RadarSINR_hist = zeros(MaxIter,1);
    ULSINR_hist    = zeros(MaxIter, params.Pnum);
    DLSINR_hist    = zeros(MaxIter, params.M);

    % ============================================================
    % 4) MAIN LOOP
    % ============================================================
    for it = 1:MaxIter

        % mutation decreases gradually
        mut_amp = mut_amp0 * (1 - it/MaxIter) + mut_ampMin * (it/MaxIter);

        % sort population by brightness
        [~,ord] = sort([Pop.B], 'descend');
        Pop = Pop(ord);

        NewPop = Pop(1:eliteCount);
        k = eliteCount + 1;

        % generate offspring
        while k <= Npop
        
            p1 = tournament_select(Pop, tournamentK);
            p2 = tournament_select(Pop, tournamentK);
        
            if p2 == p1 && Npop > 1
                p2 = mod(p1, Npop) + 1;
            end
        
            X1 = Pop(p1).X;
            X2 = Pop(p2).X;
        
            if rand < pc
                [C1, C2] = crossover_isac(X1, X2, params);
            else
                C1 = X1;
                C2 = X2;
            end
        
            C1 = mutate_isac(C1, params, mut_amp, pm);
            C2 = mutate_isac(C2, params, mut_amp, pm);
        
            C1 = refresh_solution_state(C1, params);
            C2 = refresh_solution_state(C2, params);
        
            B1 = brightness(C1, params);
            B2 = brightness(C2, params);
        
            child.X = C1;
            child.B = B1;
            child.Pbest = C1;
            child.Bbest = B1;
        
            NewPop(k) = child;
            k = k + 1;
        
            if k <= Npop
                child.X = C2;
                child.B = B2;
                child.Pbest = C2;
                child.Bbest = B2;
        
                NewPop(k) = child;
                k = k + 1;
            end
        
        end

        % trim if overshoot
        NewPop = NewPop(1:Npop);

        % update personal/global best
        for i = 1:Npop
            if NewPop(i).B > NewPop(i).Bbest
                NewPop(i).Pbest = NewPop(i).X;
                NewPop(i).Bbest = NewPop(i).B;
            end
            if NewPop(i).B > Best.B
                Best = NewPop(i);
            end
        end

        Pop = NewPop;
        B_history(it) = Best.B;

        % ============================================================
        % RECORD SINR OF CURRENT BEST
        % ============================================================
        Xb = Best.X;

        RadarSINR_hist(it) = Compute_Radar_SINR( ...
            Xb.w, Xb.Vc, Xb.Vs, ...
            params.theta_target, ...
            params.theta_clutter, params.alpha_clutter, ...
            params.alpha_t, ...
            Xb.rho, ...
            params.H_ul, ...
            params.H_SI, ...          
            params.sigma2, ...        % radar noise (DL noise)
            params.steer);

        UL_tmp = zeros(params.Pnum,1);
        for u = 1:params.Pnum
            UL_tmp(u) = compute_UL_SINR( ...
                Xb.U{u}, params.Nt, params.Nr, params.H_ul, ...
                Xb.Vc, Xb.Vs, Xb.rho, ...
                params.sigma_u2, ...
                params.alpha_t, params.theta_target, ...
                params.alpha_clutter, params.theta_clutter, ...
                params.H_SI, ...  
                u, ...
                params.steer);
        end
        ULSINR_hist(it,:) = UL_tmp;

        DL_tmp = zeros(params.M,1);
        for m = 1:params.M
            DL_tmp(m) = compute_DL_SINR( ...
                params.h_dl(:,m), ...
                Xb.Vc, Xb.Vs, Xb.rho, ...
                params.g_m(:,m), params.sigma2, m);
        end
        DLSINR_hist(it,:) = DL_tmp;
    end

    % ============================================================
    % 6) RETURN
    % ============================================================
    result.Best           = Best;
    result.B_history      = B_history;
    result.RadarSINR_hist = RadarSINR_hist;
    result.ULSINR_hist     = ULSINR_hist;
    result.DLSINR_hist     = DLSINR_hist;
    result.params         = params;
end

% ============================================================
% TOURNAMENT SELECTION
% ============================================================
function idx = tournament_select(Pop, K)
    n = numel(Pop);
    cand = randi(n, [K,1]);
    B = arrayfun(@(s) s.B, Pop(cand));
    [~,ii] = max(B);
    idx = cand(ii);
end

% ============================================================
% CROSSOVER
% ============================================================
function [C1, C2] = crossover_isac(X1, X2, params)
    alpha = rand();  % blend factor

    C1.Vc  = alpha * X1.Vc  + (1-alpha) * X2.Vc;
    C2.Vc  = alpha * X2.Vc  + (1-alpha) * X1.Vc;

    C1.Vs = zeros(size(X1.Vs));
    C2.Vs = zeros(size(X2.Vs));

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
        X.rho = X.rho + mut_amp * randn(size(X.rho));
        X.rho = min(max(X.rho,0),params.P_UL);
    end
end

function X = refresh_solution_state(X, params)

    eps0 = 1e-12;

    % =========================================================
    % 0) INIT U và w nếu chưa có
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

        % ===== Gp =====
        Gp = zeros(params.Nr, params.Nr);

        % ----- UL multi-user interference -----
        for i = 1:params.Pnum
            if i ~= pp
                hi = params.H_ul(:,i);
                Gp = Gp + X.rho(i) * (hi * hi');
            end
        end

        % ----- (b) TARGET (k = 0) -----
        A_t = params.A_target;
        Gp = Gp + abs(params.alpha_t)^2 * (A_t * Rtx * A_t');

        % ----- (c) CLUTTER -----
        for k = 1:length(params.theta_clutter)
            A_c = params.A_clutter(:,:,k);
            Gp = Gp + abs(params.alpha_clutter(k))^2 * (A_c * Rtx * A_c');
        end

        % ----- (d) SELF-INTERFERENCE -----
        Gp = Gp + params.H_SI * Rtx * params.H_SI';

        % ----- (e) NOISE -----
        Gp = Gp + params.sigma_u2 * eye(params.Nr);

        % ===== MVDR update =====
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

    % ----- (1) CLUTTER -----
    for k = 1:length(params.theta_clutter)
        A_c = params.A_clutter(:,:,k);
        Rrad = Rrad + abs(params.alpha_clutter(k))^2 * (A_c * Rtx * A_c');
    end

    % ----- (2) UL interference -----
    for pp = 1:params.Pnum
        hp = params.H_ul(:,pp);
        Rrad = Rrad + X.rho(pp) * (hp * hp');
    end

    % ----- (3) SELF-INTERFERENCE -----
    Rrad = Rrad + params.H_SI * Rtx * params.H_SI';

    % ===== MVDR =====
    temp = Rrad \ a_r;
    X.w = temp / (a_r' * temp + eps0);

end
