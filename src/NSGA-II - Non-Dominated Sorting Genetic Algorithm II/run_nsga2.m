clear; close all; clc;
seed = 150;
rng(seed);

%% ==========================================================
% 0) SYSTEM PARAMETERS
%% ==========================================================
N = 16;
Nt = N;
Nr = N;
Nu = 1;

M = 2;          % number of DL users
Pnum = 2;       % number of UL users
K_clutter = 2;  % number of clutter points
K = 1 + K_clutter;   % target + clutter

eta = 0.85;

scale_factor = 1;
BW = 20e6;
fc = 2e9;
c = 3e8;
wavlen = c/fc;
No_dBmHz = -174;

Pd_dBm = 46;
P_DL = 10^((Pd_dBm - 30)/10);

Pu_dBm = 5;
P_UL = 10^((Pu_dBm - 30)/10);

NF_dl = 9;
NF_ul = 9;

sigma2 = 10^((No_dBmHz + 10*log10(BW) + NF_dl - 30)/10) * scale_factor^2;   % DL noise
sigma_u2 = 10^((No_dBmHz + 10*log10(BW) + NF_ul - 30)/10) * scale_factor^2; % UL noise

Gamma_r      = 10^(10/10);

lambda_comm  = 1;
lambda_DL = 0.001;
lambda_UL = 0.002;
lambda_radar = 1;
lambda_pow   = 0.001;

%% ==========================================================
% 1) GEOMETRY
%% ==========================================================
BS_pos = [0, 0];

DL_angles = [-35 50];
UL_angles = [-10 20];

theta_target = 10;           % deg
theta_clutter = [-50 40]';    % deg

d_DL = [50 60];
d_UL = [40 50];
d_tag = 50;

Pos_DL = [d_DL.*cosd(DL_angles); d_DL.*sind(DL_angles)].';
Pos_UL = [d_UL.*cosd(UL_angles); d_UL.*sind(UL_angles)].';

d_BS_DL = sqrt(sum((Pos_DL - BS_pos).^2, 2));
d_BS_UL = sqrt(sum((Pos_UL - BS_pos).^2, 2));

d_UL_DL = zeros(M, Pnum);
for m = 1:M
    for p = 1:Pnum
        d_UL_DL(m,p) = norm(Pos_DL(m,:) - Pos_UL(p,:));
    end
end

steer = @(theta,N) ...
    exp(1j*pi*((1:N).' - N/2)*sind(theta)) / sqrt(N);

%% ==========================================================
% 2) CHANNEL
%% ==========================================================
PL_lin_DL = 10.^(-(128.1 + 37.6*log10(d_BS_DL/1000))/10);
H_DL = zeros(M, N);
for m = 1:M
    H_DL(m,:) = scale_factor * sqrt(PL_lin_DL(m)) * ...
        (randn(1,N) + 1j*randn(1,N))/sqrt(2);
end

PL_lin_UL = 10.^(-(128.1 + 37.6*log10(d_BS_UL/1000))/10);
H_UL = zeros(N, Pnum);
for p = 1:Pnum
    H_UL(:,p) = scale_factor * sqrt(PL_lin_UL(p)) * ...
        (randn(N,1) + 1j*randn(N,1))/sqrt(2);
end

PL_lin_CCI = 10.^(-(128.1 + 37.6*log10(d_UL_DL/1000))/10);
G_CCI = zeros(M, Pnum);
for m = 1:M
    for p = 1:Pnum
        G_CCI(m,p) = scale_factor * sqrt(PL_lin_CCI(m,p)) * ...
            (randn + 1j*randn)/sqrt(2);
    end
end

%% ==========================================================
% 3) TARGET & CLUTTER 
%% ==========================================================
sigma_target = 1;
path_loss_radar = (wavlen^2 * sigma_target) / ((4*pi)^3 * d_tag^4);
alpha_t = scale_factor * sqrt(path_loss_radar) * exp(1j*rand*2*pi);

sigma_clutter = 10;
dist_clutter = [30 50]';

alpha_clutter = zeros(K_clutter,1);
for k = 1:K_clutter
    path_loss_clutter = (wavlen^2 * sigma_clutter) / ((4*pi)^3 * dist_clutter(k)^4);
    alpha_clutter(k) = scale_factor * sqrt(path_loss_clutter) * exp(1j*rand*2*pi);
end

a_t = steer(theta_target, Nt);
a_r = steer(theta_target, Nr);
A_target = a_r * a_t';

A_clutter = zeros(Nr, Nt, K_clutter);
for k = 1:K_clutter
    a_tk = steer(theta_clutter(k), Nt);
    a_rk = steer(theta_clutter(k), Nr);
    A_clutter(:,:,k) = a_rk * a_tk';
end

%% ==========================================================
% 4) PACK INTO PARAMS
%% ==========================================================
params = struct();

params.Nt = Nt;
params.Nr = Nr;
params.Nu = Nu;
params.M = M;
params.Pnum = Pnum;
params.K = K;

params.P_DL = P_DL;
params.P_UL = P_UL;
params.mu = mu;
params.eta = eta;

params.sigma2 = sigma2;
params.sigma_u2 = sigma_u2;

params.Gamma_r = Gamma_r;

params.lambda_radar = lambda_radar;
params.lambda_pow = lambda_pow;
params.lambda_comm = lambda_comm;
params.lambda_DL = lambda_DL;
params.lambda_UL = lambda_UL;

params.h_dl = H_DL.';      % Nt x M
params.H_ul = H_UL;        % Nr x Pnum
params.g_m  = G_CCI.';     % Pnum x M

params.theta_target = theta_target;
params.theta_clutter = theta_clutter;
params.A_target = A_target;
params.A_clutter = A_clutter;
params.alpha_t = alpha_t;
params.alpha_clutter = alpha_clutter;

params.steer = steer;
params.a_t = a_t;
params.a_r = a_r;
params.DL_angles = DL_angles;
params.UL_angles = UL_angles;

params.BS_pos = BS_pos;
params.Pos_DL = Pos_DL;
params.Pos_UL = Pos_UL;

%% ==========================================================
% 5) RUN NSGA-II
%% ==========================================================
disp('Running NSGA-II...');

res = NSGA2_ISAC_function(params, seed);

Pop = res.Pop;

Obj = vertcat(Pop.Obj);

f1 = 10*log10(-Obj(:,1) + 1e-12);
f2 = 10*log10(-Obj(:,2) + 1e-12);


figure;

scatter(f1, f2, 60, 'filled');

grid on;
xlabel('Minimum UL SINR (dB)');
ylabel('Minimum DL SINR (dB)');
title('Final Pareto Front');

figure;
hold on;
grid on;

step = max(1, floor(length(res.Obj_history)/10));

for k = 1:step:length(res.Obj_history)

    Obj = res.Obj_history{k};

    UL = 10*log10(-Obj(:,1) + 1e-12);
    DL = 10*log10(-Obj(:,2) + 1e-12);

    scatter(UL, DL, 20);

end

xlabel('Minimum UL SINR (dB)');
ylabel('Minimum DL SINR (dB)');
title('Pareto Front Evolution');

Rank = [Pop.Rank];

figure;
hold on;
grid on;

idx = (Rank == 1);

scatter(f1(~idx), f2(~idx), 40, 'k');
scatter(f1(idx),  f2(idx),  80, 'filled');

xlabel('Minimum UL SINR (dB)');
ylabel('Minimum DL SINR (dB)');
title('Rank-1 Pareto Solutions');

legend('Dominated','Pareto','Location','best');


fprintf('\n=========================================\n');
fprintf('          FINAL PARETO SOLUTIONS\n');
fprintf('=========================================\n');

idx = find([Pop.Rank] == 1);

for i = 1:length(idx)

    X = Pop(idx(i)).X;

    fprintf('\n=================================================\n');
    fprintf('Pareto Solution %d\n', i);
    fprintf('=================================================\n');

    %--------------------------------------------------
    % Objectives
    %--------------------------------------------------
    fprintf('Minimum UL SINR = %.3f\n', 10*log10(-Pop(idx(i)).Obj(1) + 1e-12));
    fprintf('Minimum DL SINR = %.3f\n', 10*log10(-Pop(idx(i)).Obj(2) + 1e-12));

    %--------------------------------------------------
    % Radar SINR
    %--------------------------------------------------
    RadarSINR = Compute_Radar_SINR( ...
        X.w, X.Vc, X.Vs, ...
        params.theta_target, ...
        params.theta_clutter, ...
        params.alpha_clutter, ...
        params.alpha_t, ...
        X.rho, ...
        params.H_ul, ...
        params.sigma2, ...
        params.steer);

    fprintf('Radar SINR = %.3f\n', 10*log10(RadarSINR + 1e-12));

    %--------------------------------------------------
    % Individual UL SINR
    %--------------------------------------------------
    fprintf('\nUL SINR:\n');
    for u = 1:params.Pnum

        UL = compute_UL_SINR( ...
            X.U{u}, params.Nt, params.Nr, params.H_ul, ...
            X.Vc, X.Vs, X.rho, ...
            params.sigma_u2, ...
            params.alpha_t, params.theta_target, ...
            params.alpha_clutter, params.theta_clutter, ...
            u, ...
            params.steer);

        fprintf('  User %d : %.3f\n', u, 10*log10(UL + 1e-12));

    end

    %--------------------------------------------------
    % Individual DL SINR
    %--------------------------------------------------
    fprintf('\nDL SINR:\n');
    for m = 1:params.M

        DL = compute_DL_SINR( ...
            params.h_dl(:,m), ...
            X.Vc, X.Vs, X.rho, ...
            params.g_m(:,m), ...
            params.sigma2, ...
            m);

        fprintf('  User %d : %.3f\n', m, 10*log10(DL + 1e-12));

    end
end