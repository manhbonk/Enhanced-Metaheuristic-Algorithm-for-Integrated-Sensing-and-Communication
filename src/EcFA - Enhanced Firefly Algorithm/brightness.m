function F = brightness(X, p)

    eps0 = 1e-12;
    
    % ================= TX covariance =================
    R = X.Vc * X.Vc' + X.Vs * X.Vs';
    P_tot = real(trace(R)) + eps0;
    
    % ================= DL SINR =================
    DLs = zeros(p.M, 1);
    for m = 1:p.M
        DLs(m) = compute_DL_SINR( ...
            p.h_dl(:,m), X.Vc, X.Vs, X.rho, ...
            p.g_m(:,m), p.sigma2, m);
    end
    
    % ================= UL SINR =================
    ULs = zeros(p.Pnum, 1);
    for n = 1:p.Pnum
        ULs(n) = compute_UL_SINR( ...
            X.U{n}, p.Nt, p.Nr, p.H_ul, ...
            X.Vc, X.Vs, X.rho, ...
            p.sigma_u2, ...
            p.alpha_t, p.theta_target, ...
            p.alpha_clutter, p.theta_clutter, ...
            p.H_SI, ...
            n, ...
            p.steer);
    end
    
    % ================= Radar SINR =================
    SINR_radar = Compute_Radar_SINR( ...
        X.w, X.Vc, X.Vs, ...
        p.theta_target, ...
        p.theta_clutter, p.alpha_clutter, ...
        p.alpha_t, ...
        X.rho, ...
        p.H_ul, ...
        p.H_SI, ...
        p.sigma2, ...
        p.steer);
 
    % =========================================================
    % COMMUNICATION
    % =========================================================
    comm_vec = [
        p.mu * min(ULs);
        (1 - p.mu) * min(DLs)
    ];
    
    F_comm = smooth_min(comm_vec, 0.3);

%     min_UL = 2 * min(ULs);
%     min_DL = min(DLs);
%     
%     F_comm = sqrt(min_UL * min_DL);

    % =========================================================
    % SINR PENALTY 
    % =========================================================
    mean_DL = mean(DLs);
    mean_UL = mean(ULs);
    
    pen_DL = max((DLs - mean_DL).^2);
    pen_UL = max((ULs - mean_UL).^2);

    pen_radar = max(0, p.Gamma_r - SINR_radar).^2;
    
    % =========================================================
    % POWER PENALTY
    % =========================================================
    pen_power = max(0, (P_tot / (p.P_DL + eps0)) - 1)^2;

    % =========================================================
    % FINAL OBJECTIVE
    % =========================================================
    F = p.lambda_comm    * F_comm ...
      - p.lambda_pow     * pen_power ...
      - p.lambda_radar   * pen_radar ...
      - p.lambda_DL * pen_DL - p.lambda_UL * pen_UL;

    F = real(F);
end

function y = smooth_min(x, tau)

    x = real(x(:));

    % numerical stabilization
    xmin = min(x);

    y = xmin ...
      - (1/tau) * log(mean(exp(-tau * (x - xmin))));

end
