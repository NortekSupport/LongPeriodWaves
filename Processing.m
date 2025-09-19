function [IG, Pressure, AST] = ProcessIGW_Nortek(time, Fs, varargin)

% This code uses AST and pressure data from Nortek instruments (Signature, 
% AWAC and Aquadopps) to calculate IG waves from 25 to 300 s wave periods, 
% adjustable on line 25. For Support related-questions refer to 
% support@nortekgroup.com

%% Data Loading

p = inputParser;
addParameter(p, 'pressure', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'ast', [], @(x) isnumeric(x) || isempty(x));
parse(p, varargin{:});
pressure_raw = p.Results.pressure;
ast_raw      = p.Results.ast;

if isempty(pressure_raw) && isempty(ast_raw)
    error('Provide at least one sensor: ''pressure'', pressure_raw and/or ''ast'', ast_raw.');
end

% === Parameters ===
samplesInHour = 3600*Fs;

% define cutoff frequencies
IG_band       = [1/250, 1/25];   % IG frequency bounds [Hz]
SeaSwell_band = [1/25,  1/5];    % Sea/Swell band [Hz]  (change if needed)

IG = struct(); Pressure = struct(); AST = struct();
IG.Time_hour = time(1:samplesInHour:end);
IG.time=time;
%% Split by available data
if ~isempty(ast_raw) && ~isempty(pressure_raw)
    % Spectral
    [IG, Pressure, AST] = run_spectral_analysis(pressure_raw, ast_raw, IG_band, samplesInHour, Fs, IG);
    % hourly time vectors
    if ~isempty(time)
        if isfield(Pressure,'Spectral') && isfield(Pressure.Spectral,'SB')
            nP = size(Pressure.Spectral.SB,2);
            Pressure.time_hourly = time(1:samplesInHour:nP*samplesInHour);
        end
        if isfield(AST,'Spectral') && isfield(AST.Spectral,'SB')
            nA = size(AST.Spectral.SB,2);
            AST.time_hourly = time(1:samplesInHour:nA*samplesInHour);
        end
    end

    % ZDC
    [IG, Pressure, AST] = run_zdc_analysis(IG, Pressure, AST, pressure_raw, ast_raw, IG_band, Fs, samplesInHour);

    % Wavelet
    IG = run_wavelet_analysis(IG, pressure_raw, ast_raw, IG_band, samplesInHour, Fs);

elseif ~isempty(ast_raw)
    % AST only
    [IG, ~, AST] = run_spectral_analysis([], ast_raw, IG_band, samplesInHour, Fs, IG);
    if ~isempty(time) && isfield(AST,'Spectral') && isfield(AST.Spectral,'SB')
        nA = size(AST.Spectral.SB,2);
        AST.time_hourly = time(1:samplesInHour:nA*samplesInHour);
    end
    [IG, ~, AST] = run_zdc_analysis(IG, struct(), AST, [], ast_raw, IG_band, Fs, samplesInHour);
    IG = run_wavelet_analysis(IG, [], ast_raw, IG_band, samplesInHour, Fs);

elseif ~isempty(pressure_raw)
    % Pressure only
    [IG, Pressure, ~] = run_spectral_analysis(pressure_raw, [], IG_band, samplesInHour, Fs, IG);
    if ~isempty(time) && isfield(Pressure,'Spectral') && isfield(Pressure.Spectral,'SB')
        nP = size(Pressure.Spectral.SB,2);
        Pressure.time_hourly = time(1:samplesInHour:nP*samplesInHour);
    end
    [IG, Pressure, ~] = run_zdc_analysis(IG, Pressure, struct(), pressure_raw, [], IG_band, Fs, samplesInHour);
    IG = run_wavelet_analysis(IG, pressure_raw, [], IG_band, samplesInHour, Fs);
end

%% ======================= Function: Spectral Analysis =======================
    function [IG, Pressure, AST] = run_spectral_analysis(pressure_raw, ast_raw, IG_band, samplesInHour, Fs, IG)
        Pressure = struct(); AST = struct();

        % ---- AST branch (if provided) ----
        if ~isempty(ast_raw)
            numSegmentsA = floor(length(ast_raw)/samplesInHour);
            for i = 1:numSegmentsA
                idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
                astSegment = ast_raw(idx);

                [AST.Spectral.Hm0_hour(i), AST.Spectral.Tm01_hour(i), AST.Spectral.Tm02_hour(i), ...
                    AST.Spectral.Tp_hour(i), ~, AST.Spectral.freq(:,i), AST.Spectral.SB(:,i)] = ...
                    WaveSpectraFun(astSegment, Fs, samplesInHour./Fs, 2^12, nanmean(astSegment),...
                    0.65, 1/25, 1/5, 'off', -3, 'off', 'off', 'off', 'off');
            end

            % IG-band moments from AST spectra
            for i = 1:size(AST.Spectral.SB,2)
                deltaF = nanmean(diff(AST.Spectral.freq(:,i)));
                inIG   = (AST.Spectral.freq(:,i) >= IG_band(1) & AST.Spectral.freq(:,i) <= IG_band(2));
                fIG    = AST.Spectral.freq(inIG,i);
                SIG    = AST.Spectral.SB(inIG,i);

                m0 = nansum(SIG) * deltaF;
                m1 = nansum(fIG .* SIG) * deltaF;
                m2 = nansum((fIG.^2) .* SIG) * deltaF;

                IG.Spectral.AST.Hm0(i)  = 4 * sqrt(m0);
                IG.Spectral.AST.Tm01(i) = m0 / m1;
                IG.Spectral.AST.Tm02(i) = sqrt(m0 / m2);

                [~,k] = max(SIG);
                IG.Spectral.AST.Tp(i) = 1 ./ fIG(k);
            end
        end

        % ---- Pressure branch ----
        if ~isempty(pressure_raw)
            numSegmentsP = floor(length(pressure_raw)/samplesInHour);
            for i = 1:numSegmentsP
                idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
                pressSegment = pressure_raw(idx);

                % Pressure -> surface elevation
                Eta = PcorFFTFun(pressSegment, Fs, samplesInHour./Fs, 2^12, mean(pressSegment),...
                    0.65, 1/25, 1/5, 1, 'all', 'off', 'off');

                [Pressure.Spectral.Hm0_hour(i), Pressure.Spectral.Tm01_hour(i), Pressure.Spectral.Tm02_hour(i), ...
                    Pressure.Spectral.Tp_hour(i), ~, Pressure.Spectral.freq(:,i), Pressure.Spectral.SB(:,i)] = ...
                    WaveSpectraFun(Eta', Fs, samplesInHour./Fs, 2^12, mean(pressSegment),...
                    0.65, 1/25, 1/5, 'off', -3, 'off', 'off', 'off', 'off');
            end

            % IG-band moments from Pressure spectra
            for i = 1:size(Pressure.Spectral.SB,2)
                deltaF = nanmean(diff(Pressure.Spectral.freq(:,i)));
                inIG   = (Pressure.Spectral.freq(:,i) >= IG_band(1) & Pressure.Spectral.freq(:,i) <= IG_band(2));
                fIG    = Pressure.Spectral.freq(inIG,i);
                SIG    = Pressure.Spectral.SB(inIG,i);

                m0 = nansum(SIG) * deltaF;
                m1 = nansum(fIG .* SIG) * deltaF;
                m2 = nansum((fIG.^2) .* SIG) * deltaF;

                IG.Spectral.Pressure.Hm0(i)  = 4 * sqrt(m0);
                IG.Spectral.Pressure.Tm01(i) = m0 / m1;
                IG.Spectral.Pressure.Tm02(i) = sqrt(m0 / m2);

                [~,k] = max(SIG);
                IG.Spectral.Pressure.Tp(i) = 1 ./ fIG(k);
            end
        end
    end

%% ================== Function: Time-Domain Filtering + ZDC ==================
    function [IG, Pressure, AST] = run_zdc_analysis(IG, Pressure, AST, pressure_raw, ast_raw, IG_band, Fs, samplesInHour)

        bpIG = designfilt('bandpassiir', 'FilterOrder', 6, ...
            'HalfPowerFrequency1', IG_band(1), 'HalfPowerFrequency2', IG_band(2), 'SampleRate', Fs);

        bpSS = designfilt('bandpassiir', 'FilterOrder', 6, ...
            'HalfPowerFrequency1', 0.04, 'HalfPowerFrequency2', 0.2, 'SampleRate', Fs);

        % ---- AST branch ----
        if ~isempty(ast_raw)
            IG.zdc.AST.filtered_TS = filtfilt(bpIG, detrend(ast_raw));
            AST.zdc.filtered_SS    = filtfilt(bpSS, detrend(ast_raw));

            numSegmentsA = floor(length(ast_raw)/samplesInHour);
            IG.zdc.AST.Flag_AST_hour   = zeros(numSegmentsA,1);
            IG.zdc.AST.Hm0         = nan(1,numSegmentsA);
            IG.zdc.AST.Tp          = nan(1,numSegmentsA);

            for i = 1:numSegmentsA
                idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
                astSeg = IG.zdc.AST.filtered_TS(idx);

                % zero down-crossings (AST)
                crossings = find(astSeg(1:end-1) > 0 & astSeg(2:end) <= 0);
                nW = numel(crossings) - 1;
                IG.zdc.Flag_AST_hour(i) = nW > 50;

                if nW >= 1
                    H = nan(1,nW); T = nan(1,nW);
                    for j = 1:nW
                        w = astSeg(crossings(j):crossings(j+1));
                        H(j) = max(w) - min(w);
                        T(j) = (crossings(j+1) - crossings(j)) / Fs;
                    end
                    IG.zdc.AST.Hm0(i) = max(H);
                    IG.zdc.AST.Tp(i)  = max(T);
                end

                % spectral (IG band) on filtered AST segment
                [IG.zdc.AST.Hm0(i), IG.zdc.AST.Tm01(i), IG.zdc.AST.Tm02(i), ...
                    IG.zdc.AST.Tp(i), IG.zdc.AST.fp(i,:), IG.zdc.AST.f(i,:), ...
                    IG.zdc.AST.SB(i,:)] = WaveSpectraFun(astSeg, Fs, ...
                    samplesInHour./Fs, 2^10, nanmean(astSeg), 0.65, 1/250, 1/25, ...
                    'off', -3, 'off', 'off', 'off', 'off');
            end
        end

        % ---- Pressure branch ----
        if ~isempty(pressure_raw)
            IG.zdc.Pressure.filtered_TS = filtfilt(bpIG, detrend(pressure_raw));
            Pressure.zdc.filtered_SS    = filtfilt(bpSS, detrend(pressure_raw));

            numSegmentsP = floor(length(pressure_raw)/samplesInHour);
            IG.zdc.Pressure.Flag_press_hour = zeros(numSegmentsP,1);
            IG.zdc.Pressure.Hm0    = nan(1,numSegmentsP);
            IG.zdc.Pressure.Tp     = nan(1,numSegmentsP);

            for i = 1:numSegmentsP
                idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
                pSeg = IG.zdc.Pressure.filtered_TS(idx);

                % zero down-crossings (Pressure)
                crossings = find(pSeg(1:end-1) > 0 & pSeg(2:end) <= 0);
                nW = numel(crossings) - 1;
                IG.zdc.Press.Flag_press_hour(i) = nW > 50;

                if nW >= 1
                    H = nan(1,nW); T = nan(1,nW);
                    for j = 1:nW
                        w = pSeg(crossings(j):crossings(j+1));
                        H(j) = max(w) - min(w);
                        T(j) = (crossings(j+1) - crossings(j)) / Fs;
                    end
                    IG.zdc.Pressure.Hm0(i) = max(H);
                    IG.zdc.Pressure.Tp(i)  = max(T);
                end

                % spectral (IG band) on filtered Pressure segment
                [IG.zdc.Pressure.Hm0(i), IG.zdc.Pressure.Tm01(i), IG.zdc.Pressure.Tm02(i), ...
                    IG.zdc.Pressure.Tp(i), IG.zdc.Pressure.fp(i), IG.zdc.Pressure.f(i,:), ...
                    IG.zdc.Pressure.SB(i,:)] = WaveSpectraFun(pSeg, Fs, ...
                    samplesInHour./Fs, 2^10, nanmean(pSeg), ...
                    0.65, 1/250, 1/25, 'off', -3, 'off', 'off', 'off', 'off');
            end
        end
    end

%% Morlet CWT → ICWT band reconstruction (IG band) → variance → Hm0
    function IG = run_wavelet_analysis(IG, pressure_raw, ast_raw, IG_band, samplesInHour, Fs)

        % ---- AST branch ----
        if ~isempty(ast_raw)

            seg_sec = 600;                 % window length (s)
            hop_sec = 300;                 % hop (s) e.g., 50% overlap
            Nwin    = round(seg_sec*Fs);
            Nhop    = round(hop_sec*Fs);

            x = ast_raw  ;
            N = numel(x);

            % Get frequency grid once (using a dummy window)
            [~, f_ref] = cwt(x(1:min(N,Nwin)), Fs, 'amor', ...
                'FrequencyLimits', [1/300 1/5], 'VoicesPerOctave', 10);
            F = numel(f_ref);

            % Accumulators over full signal length
            cfs_accum = zeros(F, N);
            w_accum   = zeros(1, N);   % how many windows contributed to each sample
            nSegments = floor((N - Nwin) / Nhop) + 1;

            power_accum = zeros(F, N, 'single');   % stores sum of |cfs|^2
            w_accum     = zeros(1, N, 'single');   % coverage count per time index

            h = waitbar(0, 'Computing CWT segments...');

            start_idx = 1;
            seg_idx   = 0;

            while start_idx + Nwin - 1 <= N
                seg_idx = seg_idx + 1;
                idx = start_idx : start_idx + Nwin - 1;

                seg = x(idx);

                % CWT for this window
                [cfs_seg, f] = cwt(seg, Fs, 'amor', ...
                    'FrequencyLimits', [1/300 1/5], 'VoicesPerOctave', 10);

                % Safety: ensure frequency grid is identical
                if ~isequal(f, f_ref)
                    error('CWT frequency grid changed between segments.');
                end

                % ---- Accumulate POWER directly ----
                P_seg = abs(cfs_seg).^2;                   % units = (input units)^2
                power_accum(:, idx) = power_accum(:, idx) + single(P_seg);
                w_accum(idx)        = w_accum(idx) + 1;

                % Progress
                waitbar(seg_idx/nSegments, h, ...
                    sprintf('Processing segment %d of %d...', seg_idx, nSegments));

                start_idx = start_idx + Nhop;
            end
            close(h);

            % Normalize by coverage (mean power per (f,t))
            w_accum(w_accum==0) = 1;
            power_full = bsxfun(@rdivide, power_accum, w_accum);  % [F x N], Pa^2 or m^2

            % Time axis
            t_full = (0:N-1)/Fs;

            % ---- (Optional) Variance normalization ----
            % This makes values interpretable as "fraction of variance"
            varx = var(x, 'omitnan');
            power_frac = double(power_full) / varx;        % unitless

            % ---- (Optional) dB scale for plotting ----
            LP = 10*log10(double(power_full) + eps);       % dB (dimensionless)

            % ---- Save to struct ----
            IG.Wavelet.AST.f   = f_ref;
            IG.Wavelet.AST.t   = t_full;
            IG.Wavelet.AST.pow = power_full;          % (Pa^2 or m^2)
            IG.Wavelet.AST.pow_frac = power_frac;     % (unitless, fraction of var)
            IG.Wavelet.AST.pow_dB   = LP;             % (dB)
        end

        % ---- Pressure branch ----
        if ~isempty(pressure_raw)

            seg_sec = 600;                 % window length (s)
            hop_sec = 300;                 % hop (s) e.g., 50% overlap
            Nwin    = round(seg_sec*Fs);
            Nhop    = round(hop_sec*Fs);

            x = pressure_raw  ;
            N = numel(x);

            % Get frequency grid once (using a dummy window)
            [~, f_ref] = cwt(x(1:min(N,Nwin)), Fs, 'amor', ...
                'FrequencyLimits', [1/300 1/5], 'VoicesPerOctave', 16);
            F = numel(f_ref);

            % Accumulators over full signal length
            cfs_accum = zeros(F, N);
            w_accum   = zeros(1, N);   % how many windows contributed to each sample
            nSegments = floor((N - Nwin) / Nhop) + 1;

            power_accum = zeros(F, N, 'single');   % stores sum of |cfs|^2
            w_accum     = zeros(1, N, 'single');   % coverage count per time index

            h = waitbar(0, 'Computing CWT segments...');

            start_idx = 1;
            seg_idx   = 0;

            while start_idx + Nwin - 1 <= N
                seg_idx = seg_idx + 1;
                idx = start_idx : start_idx + Nwin - 1;

                seg = x(idx);

                % CWT for this window
                [cfs_seg, f] = cwt(seg, Fs, 'amor', ...
                    'FrequencyLimits', [1/300 1/5], 'VoicesPerOctave', 16);

                % Safety: ensure frequency grid is identical
                if ~isequal(f, f_ref)
                    error('CWT frequency grid changed between segments.');
                end

                % ---- Accumulate POWER directly ----
                P_seg = abs(cfs_seg).^2;                   % units = (input units)^2
                power_accum(:, idx) = power_accum(:, idx) + single(P_seg);
                w_accum(idx)        = w_accum(idx) + 1;

                % Progress
                waitbar(seg_idx/nSegments, h, ...
                    sprintf('Processing segment %d of %d...', seg_idx, nSegments));

                start_idx = start_idx + Nhop;
            end
            close(h);

            % Normalize by coverage (mean power per (f,t))
            w_accum(w_accum==0) = 1;
            power_full = bsxfun(@rdivide, power_accum, w_accum);  % [F x N], Pa^2 or m^2

            % Time axis
            t_full = (0:N-1)/Fs;

            % ---- (Optional) Variance normalization ----
            % This makes values interpretable as "fraction of variance"
            varx = var(x, 'omitnan');
            power_frac = double(power_full) / varx;        % unitless

            % ---- (Optional) dB scale for plotting ----
            LP = 10*log10(double(power_full) + eps);       % dB (dimensionless)

            % ---- Save to struct ----
            IG.Wavelet.Pressure.f   = f_ref;
            IG.Wavelet.Pressure.t   = t_full;
            IG.Wavelet.Pressure.pow = power_full;          % (Pa^2 or m^2)
            IG.Wavelet.Pressure.pow_frac = power_frac;     % (unitless, fraction of var)
            IG.Wavelet.Pressure.pow_dB   = LP;             % (dB)
        end
    end
end
