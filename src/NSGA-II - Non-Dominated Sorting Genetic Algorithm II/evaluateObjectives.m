function Obj = evaluateObjectives(X, p)

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
            n, ...
            p.steer);
    end
    

    % ==========================================================
    % NSGA-II assumes minimization
    % ==========================================================
    Obj = [-min(ULs), -min(DLs)];

end