function sinr = compute_DL_SINR(h, Vc, Vs, rho, g, sigma2, m)
    % DL SINR of user m
    % Paper: SINR_DL_m = |h_m^H v_{c,m}|^2 / (MUI + radar + CCI + noise)

    eps0 = 1e-12;

    % desired signal
    sig = abs(h' * Vc(:, m))^2;

    % multi-user interference from other DL beams
    mui = sum(abs(h' * Vc).^2) - sig;

    % radar waveform leakage
    rad = real(h' * (Vs * Vs') * h);

    % uplink-to-downlink co-channel interference
    cci = sum(rho(:) .* abs(g(:)).^2);

    % noise
    noise = sigma2;

    sinr = real(sig / (mui + rad + cci + noise + eps0));
end
