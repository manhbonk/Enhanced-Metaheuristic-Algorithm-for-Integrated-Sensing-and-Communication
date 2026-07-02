function sinr_r = Compute_Radar_SINR( ...
    w, Vc, Vs, theta_target, theta_clutter, alpha_clutter, alpha_t, ...
    rho, H_ul, sigma2, steer)

    eps0 = 1e-12;

    Nr = length(w);
    Nt = size(Vc,1);

    % ===== Covariance =====
    R_total = Vc*Vc' + Vs*Vs';

    % ===== Steering target =====
    a_t = steer(theta_target, Nt);
    a_r = steer(theta_target, Nr);
    A_t = a_r * a_t';

    % =========================================================
    % 1) SIGNAL (ONLY sensing part Vs)
    % =========================================================
    sig = (abs(alpha_t)^2) * real( w' * (A_t * R_total * A_t') * w );

    % =========================================================
    % 2) CLUTTER
    % =========================================================
    clutter_intf = 0;
    for k = 1:length(theta_clutter)
        a_tc = steer(theta_clutter(k), Nt);
        a_rc = steer(theta_clutter(k), Nr);
        A_c = a_rc * a_tc';

        clutter_intf = clutter_intf + ...
            (abs(alpha_clutter(k))^2) * ...
            real( w' * (A_c * R_total * A_c') * w );
    end

    % =========================================================
    % 3) UL INTERFERENCE
    % =========================================================
    ul_intf = 0;
    for p = 1:length(rho)
        hp = H_ul(:, p);
        ul_intf = ul_intf + rho(p) * real( w' * (hp * hp') * w );
    end

    % =========================================================
    % 4) NOISE
    % =========================================================
    noise = sigma2 * real(w' * w);

    % =========================================================
    % FINAL SINR
    % =========================================================
    denom = clutter_intf + ul_intf + noise + eps0;

    sinr_r = real(sig / denom);
end