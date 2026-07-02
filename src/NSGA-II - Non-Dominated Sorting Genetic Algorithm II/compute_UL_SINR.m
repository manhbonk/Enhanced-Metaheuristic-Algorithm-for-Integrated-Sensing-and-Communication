function sinr = compute_UL_SINR( ...
    u, Nt, Nr, H, Vc, Vs, rho, sigma_u2, ...
    alpha_t, theta_target, alpha_clutter, theta_clutter, ...
    p, steer)

    eps0 = 1e-12;

    Pnum = length(rho);
    h_p = H(:, p);

    % =========================================================
    % 1) Desired signal
    % =========================================================
    sig = rho(p) * abs(u' * h_p)^2;

    % =========================================================
    % 2) TOTAL TX covariance
    % =========================================================
    R = Vc*Vc' + Vs*Vs';

    % =========================================================
    % 3) Build interference covariance Gp
    % =========================================================
    Gp = zeros(Nr, Nr);

    % ===== (a) UL multi-user interference =====
    for i = 1:Pnum
        if i ~= p
            hi = H(:, i);
            Gp = Gp + rho(i) * (hi * hi');
        end
    end

    % ===== (b) Target (k = 0) =====
    a_t = steer(theta_target, Nt);
    a_r = steer(theta_target, Nr);
    A_t = a_r * a_t';

    Gp = Gp + (abs(alpha_t)^2) * (A_t * R * A_t');

    % ===== (c) Clutter =====
    for k = 1:length(theta_clutter)
        a_tc = steer(theta_clutter(k), Nt);
        a_rc = steer(theta_clutter(k), Nr);
        A_c = a_rc * a_tc';

        Gp = Gp + (abs(alpha_clutter(k))^2) * (A_c * R * A_c');
    end

    % ===== (d) Noise =====
    Gp = Gp + sigma_u2 * eye(Nr);

    % =========================================================
    % 4) SINR
    % =========================================================
    sinr = real( sig / (real(u' * Gp * u) + eps0) );

    den = real(u' * Gp * u);
end