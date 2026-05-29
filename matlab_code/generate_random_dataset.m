% generate_random_dataset.m
% Generate 80K pure-random samples (Part A logic only, no GA/stratified/boundary).
% For ablation on lower-quality data: does architecture matter more?
clc; clear; close all;

%% ---- Config ----
N_SAMPLES   = 80000;
OUT_DIR     = '..\data';
OUT_SL      = fullfile(OUT_DIR, 'random_dataset_slN.txt');
OUT_ALPHA   = fullfile(OUT_DIR, 'random_dataset_alphaM.txt');

nlayer = 2;  ncell = 4;  nw = 2;  delta = 1.0;  H = 49.7;  W = 49.7;  nh = 2;
TARGET_W_SUM = W - (nw+1)*delta;   % 46.7
TARGET_H_SUM = H - (nh+1)*delta;   % 46.7
FOUR_DELTA   = 4*delta;
SAFE_MARGIN_TD = 1.5*delta;
Z_MARGIN     = 3.0;

LB = [1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,  10,10,10,10,  15,15,15,15,  57,  15,15,  0,0,0,0];
UB = [200,200,200,200,200,200,200,200,  30,30,30,30,30,30,30,30,  50,50,50,50,  40,40,40,40,  57,  40,40,  200,200,200,200];

f0   = 50:10:2000;  fL = length(f0);
mnum = 500;  Norder = 9;

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
rng(42);

%% ---- Generate parameters ----
fprintf('Generating %d random samples...\n', N_SAMPLES);
slN = zeros(N_SAMPLES, 31);
rejected = 0;
tic;
parfor i = 1:N_SAMPLES
    [sl, rej] = gen_one(LB, UB, TARGET_W_SUM, TARGET_H_SUM, ...
        FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta);
    slN(i,:) = sl;
    rejected = rejected + rej;
end
elapsed = toc;
fprintf('Done: %.1f sec (avg %.1f rejections/sample)\n', elapsed, rejected/N_SAMPLES);

%% ---- TMM Simulation ----
fprintf('Running TMM simulation on %d samples...\n', N_SAMPLES);
alphaM = zeros(fL, N_SAMPLES);
tic;
parfor n = 1:N_SAMPLES
    try
        alpha = fun_structure_broadband_nlayer_HB_func(...
            f0, slN(n,:), ncell, nw, delta, nlayer, mnum, Norder);
        alphaM(:,n) = alpha(:);
    catch
        alphaM(:,n) = NaN(fL,1);
    end
    if mod(n,10000)==0, fprintf('  %d/%d\n', n, N_SAMPLES); end
end
fprintf('Simulation done: %.1f min\n', toc/60);

%% ---- Clean & Save ----
valid = all(isfinite(alphaM),1);
fprintf('Valid: %d / %d\n', sum(valid), N_SAMPLES);
slN = slN(valid,:); alphaM = alphaM(:,valid);

save(OUT_SL,    'slN',    '-ascii','-double');
save(OUT_ALPHA, 'alphaM', '-ascii','-double');
fprintf('Saved: %s (%d x 31)\n', OUT_SL, size(slN,1));
fprintf('Saved: %s (%d x %d)\n', OUT_ALPHA, size(alphaM,1), size(alphaM,2));

%% ======== Helper functions ========
function [sl, rej] = gen_one(LB, UB, TARGET_W_SUM, TARGET_H_SUM, ...
    FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    rej = 0;
    for attempt = 1:500
        sl = zeros(1,31);
        sl(21:24) = LB(21:24) + (UB(21:24)-LB(21:24)).*rand(1,4);
        sl(25) = 57;
        [sl(17),sl(18)] = pair_sum(LB(17),UB(17),TARGET_H_SUM);
        [sl(19),sl(20)] = pair_sum(LB(19),UB(19),TARGET_H_SUM);
        [sl(26),sl(27)] = pair_sum(LB(26),UB(26),TARGET_W_SUM);

        h4 = [sl(17);sl(18);sl(19);sl(20)];
        h8 = [h4;h4];
        ok = true;
        for i = 1:8
            mx = min(UB(i), h8(i)-SAFE_MARGIN_TD);
            if mx < LB(i), ok=false; break; end
            sl(i) = LB(i) + (mx-LB(i))*rand();
        end
        if ~ok, rej=rej+1; continue; end

        ok = true;
        for k = 1:4
            mx = sl(25) - sl(20+k) - Z_MARGIN;
            if mx < LB(27+k), ok=false; break; end
            sl(27+k) = LB(27+k) + (min(UB(27+k),mx)-LB(27+k))*rand();
        end
        if ~ok, rej=rej+1; continue; end

        w2 = [sl(26);sl(26);sl(27);sl(27);sl(26);sl(26);sl(27);sl(27)];
        d_lim = min(min(w2,h8)-FOUR_DELTA, h8*0.65);
        ok = true;
        for j = 1:8
            if d_lim(j) < LB(8+j), ok=false; break; end
            sl(8+j) = LB(8+j) + (min(UB(8+j),d_lim(j))-LB(8+j))*rand();
        end
        if ~ok, rej=rej+1; continue; end

        % Full check
        h=sl(17:20); w=sl(26:27); d=sl(9:16); Lu=sl(21:24); tu=sl(28:31); Ltot=sl(25); td=sl(1:8);
        if abs(sum(w)-TARGET_W_SUM)>0.02, rej=rej+1; continue; end
        if abs(h(1)+h(2)-TARGET_H_SUM)>0.02||abs(h(3)+h(4)-TARGET_H_SUM)>0.02, rej=rej+1; continue; end
        if any(d>min(w2,h8)-FOUR_DELTA), rej=rej+1; continue; end
        if any(Lu+tu'+Z_MARGIN>Ltot), rej=rej+1; continue; end
        if any(td>h8-SAFE_MARGIN_TD), rej=rej+1; continue; end
        if any(d./h8>0.65), rej=rej+1; continue; end
        if any(h/Ltot>0.95), rej=rej+1; continue; end
        return;
    end
    error('Cannot generate valid sample.');
end

function [v1,v2] = pair_sum(lb, ub, target)
    for attempt = 1:200
        v1 = lb + (ub-lb)*rand(); v2 = target - v1;
        if v2>=lb && v2<=ub, return; end
    end
    v1 = target/2; v2 = target/2;
end
