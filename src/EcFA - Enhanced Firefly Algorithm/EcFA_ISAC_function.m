%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WMMSE-initialized Firefly Algorithm for ISAC / DFRC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = EcFA_ISAC_function(params, seed)
    rng(seed);

    % ===================== 1. FIREFLY PARAMETERS =============================
    params.Npop = 50;
    params.MaxIter = 1000;
    eta = params.eta;
    
    % FA parameters
    beta0  = 0.02;
    gamma  = 0.1;

    % ===================== 5. INITIALIZE POPULATION ==========================
    Firefly = repmat(struct('X',[],'Pbest',[],'B',[],'Bbest',[]), params.Npop, 1);
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

    for n = 1:params.Npop
        if n < 3
            X.Vc  = Vc0;
            X.Vs  = Vs0;
            
            % Equal UL power allocation
            X.rho = params.P_UL * ones(params.Pnum,1);
        elseif n <= 25
            
            % ==========================================================
            % Slightly perturbed version of n = 1
            % ==========================================================
    
            noise_level = 0.008;
    
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
            X.Vs = randn(params.Nt, params.Nt) ...
                   + 1j*randn(params.Nt, params.Nt);
            
            X.rho = params.P_UL * rand(params.Pnum,1);
    
            X.Vc  = X.Vc / norm(X.Vc,'fro') * sqrt(eta*params.P_DL);
            X.Vs  = X.Vs / norm(X.Vs)       * sqrt((1 - eta)*params.P_DL);
        end
    
        % ===== Clamp rho =====
        X.rho = min(max(X.rho,0),params.P_UL);
    
        % ===== Refresh =====
        X = refresh_solution_state(X, params);
        Firefly(n).X     = X;
        Firefly(n).Pbest = X;
        Firefly(n).B     = brightness(X,params);
        Firefly(n).Bbest = Firefly(n).B;
    end

    [~,id] = max([Firefly.B]);
    Best = Firefly(id);
    Best_idx = id;

    B_history = zeros(params.MaxIter,1);

    % ===== SINR history for convergence analysis =====
    RadarSINR_hist = zeros(params.MaxIter,1);
    ULSINR_hist    = zeros(params.MaxIter, params.Pnum);
    DLSINR_hist    = zeros(params.MaxIter, params.M);

    % ===================== 6. MAIN FA LOOP ===================================
    for it = 1:params.MaxIter

        [~,ord] = sort([Firefly.B],'descend');

        for ii = 1:params.Npop
            i = ord(ii);
            Xi = Firefly(i).X;

        % ===== alpha schedule =====
        alpha0   = 0.005;
        alpha = alpha0 * exp(-5 * it / params.MaxIter);
        
        % ---------- FA attraction to brighter fireflies ----------
        for jj = 1:ii-1
            j = ord(jj);
        
            % ===== distance & attractiveness =====
            d = fa_distance(Xi, Firefly(j).X);
            beta = beta0 * exp(-gamma * (d^2));
        
            % ===== complex Gaussian noise =====
            noiseVc  = (randn(size(Xi.Vc)) + 1j*randn(size(Xi.Vc))) / sqrt(2);
            noiseVs  = (randn(size(Xi.Vs)) + 1j*randn(size(Xi.Vs))) / sqrt(2);
            noiseRho = randn(size(Xi.rho));
        
            % ===== optional: scale-aware noise =====
            noiseVc = noiseVc .* (abs(Xi.Vc) + 1e-12);
            noiseVs = noiseVs .* (abs(Xi.Vs) + 1e-12);
        
            % ===== update Vc =====
            Xi.Vc = Xi.Vc ...
                  + beta * (Firefly(j).X.Vc - Xi.Vc) ...
                  + alpha * noiseVc;
        
            % ===== update Vs =====
            Xi.Vs = Xi.Vs ...
                  + beta * (Firefly(j).X.Vs - Xi.Vs) ...
                  + alpha * noiseVs;
        
            % ===== update rho (real variable) =====
            Xi.rho = Xi.rho ...
                   + beta * (Firefly(j).X.rho - Xi.rho) ...
                   + alpha * noiseRho;
        
            % ===== clamp rho =====
            Xi.rho = min(max(Xi.rho, 0), params.P_UL);
        end

            % ===== EcFA: global best attraction =====
            d_best = fa_distance(Xi, Best.X);
            beta_g = beta0 * exp(-gamma * (d_best^2));

            Xi.Vc  = Xi.Vc  + beta_g*(Best.X.Vc  - Xi.Vc);
            Xi.Vs  = Xi.Vs  + beta_g*(Best.X.Vs  - Xi.Vs);
            Xi.rho = Xi.rho + beta_g*(Best.X.rho - Xi.rho);
            Xi.rho = min(max(Xi.rho, 0), params.P_UL);

            % ===== EcFA: SDSS step-size =====
            delta = 0.00005 * (1 - it/params.MaxIter)^2;
            
            Xi.Vc  = Xi.Vc  + delta*(randn(size(Xi.Vc)) + 1j*randn(size(Xi.Vc)));
            Xi.Vs  = Xi.Vs  + delta*(randn(size(Xi.Vs)) + 1j*randn(size(Xi.Vs)));
            Xi.rho = Xi.rho + delta*randn(size(Xi.rho));
            Xi.rho = min(max(Xi.rho, 0), params.P_UL);

            Xi = refresh_solution_state(Xi, params);

            Firefly(i).X = Xi;
            Firefly(i).B = brightness(Xi, params);

            if Firefly(i).B > Firefly(i).Bbest
                Firefly(i).Pbest = Firefly(i).X;
                Firefly(i).Bbest = Firefly(i).B;
            end

            if Firefly(i).B > Best.B
                Best = Firefly(i);
                Best_idx = i;
            end
        end

        % ===== Mutation (target worst individuals) =====
        if mod(it,2) == 0           
            % sort theo brightness (giảm dần)
            [~,ord] = sort([Firefly.B],'descend');
            mut_amp = 0.006 * exp(-20 * it / params.MaxIter);
            % chọn K_worst cá thể tệ nhất
            K_worst = 8;
            worst_idx = ord(end - K_worst + 1 : end);
            
            for k = 1:K_worst
                idx = worst_idx(k);
                
                % mutation 
                Firefly(idx).X.Vc = Firefly(idx).X.Vc + ...
                    mut_amp*(randn(size(Firefly(idx).X.Vc)) + 1j*randn(size(Firefly(idx).X.Vc)));
                
                Firefly(idx).X.Vs = Firefly(idx).X.Vs + ...
                    mut_amp*(randn(size(Firefly(idx).X.Vs)) + 1j*randn(size(Firefly(idx).X.Vs)));
                
                % thêm mutation cho rho 
                Firefly(idx).X.rho = Firefly(idx).X.rho + ...
                    mut_amp*randn(size(Firefly(idx).X.rho));
                
                % clamp rho
                Firefly(idx).X.rho = min(max(Firefly(idx).X.rho,0),params.P_UL);
                
                % refresh 
                Firefly(idx).X = refresh_solution_state(Firefly(idx).X, params);
                
                % đánh giá lại
                Firefly(idx).B = brightness(Firefly(idx).X, params);
                
                % update pbest
                if Firefly(idx).B > Firefly(idx).Bbest
                    Firefly(idx).Pbest = Firefly(idx).X;
                    Firefly(idx).Bbest = Firefly(idx).B;
                end
                
                % update global best
                if Firefly(idx).B > Best.B
                    Best = Firefly(idx);
                    Best_idx = idx;
                end
            end
        end

        Xb = Best.X;

        % ===== EcFA: CULS local search =====
        step = 0.0005 * exp(-2*it/params.MaxIter);

        Xtest = Xb;
        Xtest.Vc = Xtest.Vc + step*(randn(size(Xb.Vc)) + 1j*randn(size(Xb.Vc)));
        Xtest.Vs = Xtest.Vs + step*(randn(size(Xb.Vs)) + 1j*randn(size(Xb.Vs)));
        Xtest.rho = Xb.rho + step*randn(size(Xb.rho));
        Xtest.rho = min(max(Xtest.rho,0),params.P_UL);

        Xtest = refresh_solution_state(Xtest, params);
        Btest = brightness(Xtest, params);

        if Btest > Best.B
            Best.X = Xtest;
            Best.B = Btest;
        
            Firefly(Best_idx).X = Xtest;
            Firefly(Best_idx).B = Btest;
        end

        Xb = Best.X;
        Firefly(Best_idx).X = Best.X;
        Firefly(Best_idx).B = Best.B;

        B_history(it) = Best.B;

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

    % ===================== RETURN RESULTS ====================================
    result.Best           = Best;
    result.B_history      = B_history;
    result.RadarSINR_hist = RadarSINR_hist;
    result.ULSINR_hist    = ULSINR_hist;
    result.DLSINR_hist    = DLSINR_hist;
    result.params         = params;
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

        % ----- (a) UL multi-user interference -----
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

    a_r = params.a_r;   % đã precompute

    Rrad = params.sigma2 * eye(params.Nr);

    % ----- (1) COMM-induced target interference (Vc ONLY) -----
    A_t = params.A_target;
    Rrad = Rrad + abs(params.alpha_t)^2 * (A_t * (X.Vc*X.Vc') * A_t');

    % ----- (2) CLUTTER -----
    for k = 1:length(params.theta_clutter)
        A_c = params.A_clutter(:,:,k);
        Rrad = Rrad + abs(params.alpha_clutter(k))^2 * (A_c * Rtx * A_c');
    end

    % ----- (3) UL interference -----
    for pp = 1:params.Pnum
        hp = params.H_ul(:,pp);
        Rrad = Rrad + X.rho(pp) * (hp * hp');
    end

    % ----- (4) SELF-INTERFERENCE -----
    Rrad = Rrad + params.H_SI * Rtx * params.H_SI';

    % ===== MVDR =====
    temp = Rrad \ a_r;
    X.w = temp / (a_r' * temp + eps0);

end
function d = fa_distance(X1, X2)
    d = norm(X1.Vc*X1.Vc' - X2.Vc*X2.Vc', 'fro') + ...
        norm(X1.Vs*X1.Vs' - X2.Vs*X2.Vs', 'fro') + ...
        norm(X1.rho - X2.rho);
end