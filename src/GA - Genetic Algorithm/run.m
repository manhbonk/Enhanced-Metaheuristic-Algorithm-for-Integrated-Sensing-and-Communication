clear; close all; clc;
seed = 10;
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

mu = 0.5;
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
lambda_UL = 0.001;
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

beta_SI_dB = -90;
beta_SI_lin = 10^(beta_SI_dB/10);
H_SI = scale_factor * sqrt(beta_SI_lin) * ...
    (randn(N,N) + 1j*randn(N,N))/sqrt(2);

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

params.H_SI = H_SI;
params.BS_pos = BS_pos;
params.Pos_DL = Pos_DL;
params.Pos_UL = Pos_UL;

%% ==========================================================
% 5) RUN SINGLE ALGORITHM
%% ==========================================================
disp('Running...');
res = EcFA_ISAC_function(params, seed);

L = length(res.B_history);

%% ==============================
% 1) Brightness
%% ==============================
figure;
plot(1:L, res.B_history,'k','LineWidth',2);
grid on;
xlabel('Iteration');
ylabel('Best Brightness');
title('Brightness Convergence');

%% ==============================
% 2) Radar SINR
%% ==============================
figure;
plot(1:L,10*log10(res.RadarSINR_hist+1e-12),'k','LineWidth',2);
grid on;
xlabel('Iteration');
ylabel('Radar SINR (dB)');
title('Radar SINR');

%% ==============================
% 3) UL SINR
%% ==============================
figure;
plot(1:L, 10*log10(res.ULSINR_hist(:,1) + 1e-12), 'r-', 'LineWidth', 2); hold on;
plot(1:L, 10*log10(res.ULSINR_hist(:,2) + 1e-12), 'b--', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('UL SINR (dB)');
title('Uplink SINR');
legend('User 1', 'User 2', 'Location', 'best');

%% ==============================
% 4) DL SINR
%% ==============================
figure;
plot(1:L, 10*log10(res.DLSINR_hist(:,1) + 1e-12), 'm-', 'LineWidth', 2); hold on;
plot(1:L, 10*log10(res.DLSINR_hist(:,2) + 1e-12), 'g--', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('DL SINR (dB)');
title('Downlink SINR');
legend('User 1', 'User 2', 'Location', 'best');

%% ==============================
% 5) Beam pattern
%% ==============================
theta = -90:0.1:90;
X = res.Best.X;

% Tx total
R = X.Vc*X.Vc' + X.Vs*X.Vs';

P_tx = zeros(size(theta));
for k = 1:length(theta)
    at = params.steer(theta(k), Nt);
    P_tx(k) = real(at' * R * at);
end
P_tx_dB = 10*log10(P_tx / max(P_tx));

% Tx sensing-only
R_s = X.Vs * X.Vs';

P_s = zeros(size(theta));
for k = 1:length(theta)
    at = params.steer(theta(k), Nt);
    P_s(k) = real(at' * R_s * at);
end
P_s_dB = 10*log10(P_s / max(P_s));

% Rx beam
wr = X.w;

P_rx = zeros(size(theta));
for k = 1:length(theta)
    ar = params.steer(theta(k), Nr);
    P_rx(k) = abs(wr' * ar)^2;
end
P_rx_dB = 10*log10(P_rx / max(P_rx));

figure;
plot(theta, P_tx_dB,'k','LineWidth',2); hold on;
plot(theta, P_s_dB,'r--','LineWidth',2);
plot(theta, P_rx_dB,'b-.','LineWidth',2);
grid on;

ymin = min([P_tx_dB P_s_dB P_rx_dB]);

plot(params.theta_target, 0, 'rp','MarkerSize',10,'LineWidth',2);
plot(params.DL_angles, zeros(size(params.DL_angles)), 'bo','LineWidth',2);
plot(params.UL_angles, zeros(size(params.UL_angles)), 'gs','LineWidth',2);
plot(theta_clutter, ymin*ones(size(theta_clutter)), 'kx','LineWidth',2);

xline(params.theta_target,'r--','LineWidth',1.5);
for t = params.DL_angles
    xline(t,'b--');
end
for t = params.UL_angles
    xline(t,'g--');
end
for t = theta_clutter'
    xline(t,'k:');
end

xlabel('Angle (deg)');
ylabel('Normalized Power (dB)');
title('Tx + Rx + Sensing Beam Pattern');
legend('Tx (Vc+Vs)', 'Tx sensing-only (Vs)', 'Rx beam (w)', ...
       'Target','DL users','UL users','Clutter');

%% ================= FINAL RESULTS =================
SINR_radar_final = res.RadarSINR_hist(end);
SINR_UL_final    = res.ULSINR_hist(end,:);
SINR_DL_final    = res.DLSINR_hist(end,:);

fprintf('\n=====================================\n');
fprintf('           FINAL RESULTS             \n');
fprintf('=====================================\n');
fprintf('Seed = %d\n', seed);

fprintf('--- Constraints ---\n');
fprintf('Gamma_r      = %.2f dB\n', 10*log10(Gamma_r));

fprintf('--- Penalty weights ---\n');
fprintf('lambda_radar = %.2f\n', lambda_radar);
fprintf('lambda_pow   = %.2f\n', lambda_pow);
fprintf('lambda_comm  = %.2f\n', lambda_comm);

fprintf('--- FINAL SINR ---\n');
fprintf('Radar SINR = %.2f dB\n', 10*log10(SINR_radar_final + 1e-12));

for i = 1:length(SINR_UL_final)
    fprintf('UL User %d SINR = %.2f dB\n', i, 10*log10(SINR_UL_final(i) + 1e-12));
end

for i = 1:length(SINR_DL_final)
    fprintf('DL User %d SINR = %.2f dB\n', i, 10*log10(SINR_DL_final(i) + 1e-12));
end