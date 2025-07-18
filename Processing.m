%% 1. Initialization and Data Loading
clear; clc;
addpath('C:\Users\NeryNeto\OneDrive - Nortek AS\Documents\IG waves\Dataset\SanMatias')

% === Parameters ===
Fs = 2;                             % Sampling frequency [Hz]
samplesInHour=3600*Fs;
% === Load Files ===
Raw = load_sanmatias_data('C:\Users\NeryNeto\OneDrive - Nortek AS\Documents\IG waves\Dataset\SanMatias');

% define cutoff frequencies 
[Eta]=PcorFFTFun(Raw.pressure(1:end-1),2,length(Raw.pressure(1:end-1)), ...
    2^15,nanmean(Raw.pressure(1:end-1)), ...
    0.65,1/250,1/2,1,'all','on','off');
% Pressure main frequencies calculation for whole period
[Pressure.Hm0_total,Pressure.Tm01_total,Pressure.Tm02_total,...
    Pressure.Tp_total,Pressure.fp_total,Pressure.f_total,...
    Pressure.SB_1_total]=WaveSpectraFun(Eta',2, ...
    length(Raw.pressure(100:end-100)),2^15,nanmean(Raw.pressure(100:end-100)), ...
    0.65,1/300,1/2,'off',-3,'on','off','off','off');
% AST main frequencies calculation for whole period
[AST.Hm0_total,AST.Tm01_total,AST.Tm02_total,...
    AST.Tp_total,AST.fp_total,AST.f_total,...
    AST.SB_1_total]=WaveSpectraFun(Raw.ast,2, ...
    length(Raw.ast),2^15,nanmean(Raw.ast),0.65,1/300,1/2,'off',-3,'on','off','on','off');
IG_band = [1/250, 1/25];           % IG frequency bounds [Hz]
SeaSwell_band = [1/25, 1/5];       % Sea/Swell band [Hz]

%% 2. Spectral Analysis (Hourly Spectra & Hm0_IG calculation)
[IG, Pressure, AST] = run_spectral_analysis(Raw, IG_band, samplesInHour);

%% 3. Time-Domain Filtering + ZDC (Zero Down-Crossing)
[IG] = run_zdc_analysis(Raw, IG_band, Fs, IG, samplesInHour);

%% 4. Wavelet Time-Frequency Analysis
IG.Hm0_wavelet = run_wavelet_analysis(Raw.ast, IG_band);


%% --- Function Definitions ---
function Raw = load_sanmatias_data(dataPath)
    fileList = dir(fullfile(dataPath, '**', '*.mat'));
    fileList = fileList(2:end-1);
    Raw.time = [];
    Raw.level = [];
    Raw.ast = [];
    Raw.pressure = [];
    Raw.astQuality = [];
    Raw.levelQuality = [];
    for k = [1 5:12 2:4]
        temp = load(fullfile(fileList(k).folder, fileList(k).name));
        Raw.time = [Raw.time; temp.Data.Burst_Time];
        Raw.level = [Raw.level; temp.Data.Burst_AltimeterDistanceLE];
        Raw.ast = [Raw.ast; temp.Data.Burst_AltimeterDistanceAST];
        Raw.pressure = [Raw.pressure; temp.Data.Burst_Pressure];
        Raw.astQuality = [Raw.astQuality; temp.Data.Burst_AltimeterQualityAST];
        Raw.levelQuality = [Raw.levelQuality; temp.Data.Burst_AltimeterQualityLE];
    end
    fields = fieldnames(Raw);
    for i = 1:numel(fields)
        Raw.(fields{i}) = Raw.(fields{i})(24998:8468930);
    end
end
%%
function [IG, Pressure, AST] = run_spectral_analysis(Raw, IG_band, samplesInHour)
    numSegments = floor(length(Raw.pressure)/samplesInHour);
    Pressure.Hm0_IG = zeros(1, numSegments);
    AST.Hm0_IG = zeros(1, numSegments);
    Pressure.Hm0_SS = zeros(1, numSegments);
    AST.Hm0_SS = zeros(1, numSegments);

    for i = 1:numSegments
        idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
        pressSegment = Raw.pressure(idx);
        astSegment = Raw.ast(idx);

        [Eta] = PcorFFTFun(pressSegment, 2, samplesInHour, 2^12, mean(pressSegment), 0.65, 1/25, 1/5, 1, 'all', 'off', 'off');
        [Pressure.Hm0_hour(i), ~, ~, ~, ~, Pressure.freq(:,i), Pressure.spectrum(:,i)] = ...
            WaveSpectraFun(Eta', 2, samplesInHour, 2^12, mean(pressSegment), 0.65, 1/25, 1/5, 'off', -3, 'off', 'off', 'off', 'off');

        [AST.Hm0_hour(i), ~, ~, ~, ~, AST.freq(:,i), AST.spectrum(:,i)] = ...
            WaveSpectraFun(astSegment, 2, samplesInHour, 2^12, mean(astSegment), 0.65, 1/25, 1/5, 'off', -3, 'off', 'off', 'off', 'off');

        deltaF = nanmean(diff(AST.freq(:,1)));
        freqIG_idx = find(Pressure.freq(:,i) >= IG_band(1) & Pressure.freq(:,i) <= IG_band(2));
        freqSS_idx = find(Pressure.freq(:,i) > IG_band(2));

        m0_IG_press = sum(Pressure.spectrum(freqIG_idx, i)) * deltaF;
        m0_IG_ast = sum(AST.spectrum(freqIG_idx, i)) * deltaF;

        IG.Spectral_press_Hm0_IG(i) = 4 * sqrt(m0_IG_press);
        IG.Spectral_AST_Hm0_IG(i) = 4 * sqrt(m0_IG_ast);

        m0_SS_press = sum(Pressure.spectrum(freqSS_idx, i)) * deltaF;
        m0_SS_ast = sum(AST.spectrum(freqSS_idx, i)) * deltaF;

        Pressure.Hm0_SS(i) = 4 * sqrt(m0_SS_press);
        AST.Hm0_SS(i) = 4 * sqrt(m0_SS_ast);
    end
    Pressure.time_hourly = Raw.time(1:samplesInHour:end); Pressure.time_hourly(end) = [];
    AST.time_hourly = Pressure.time_hourly;
end
%%
function IG = run_zdc_analysis(Raw, IG_band, Fs, IG, samplesInHour)

    bpFilter = designfilt('bandpassiir', 'FilterOrder', 6, ...
    'HalfPowerFrequency1', IG_band(1), 'HalfPowerFrequency2', IG_band(2), 'SampleRate', Fs);

%AST for full period
    signal_ast = filtfilt(bpFilter, detrend(Raw.ast));
    signal_press = filtfilt(bpFilter, detrend(Raw.pressure));

    numSegments = floor(length(Raw.pressure)/(samplesInHour)); %hourly segment
    IG.zdcFlag_AST_hour = zeros(numSegments, 1);
    IG.zdcFlag_press_hour = zeros(numSegments, 1);
    IG.TS_AST_waveHeight = [];
    IG.TS_AST_wavePeriod = [];
    IG.TS_press_waveHeight = [];
    IG.TS_press_wavePeriod = [];

    for i = 1:numSegments
        idx = (i-1)*samplesInHour + 1 : i*samplesInHour;
        pressSegment = signal_press(idx);
        astSegment = signal_ast(idx);
        % === AST ===
        crossings_ast = find(astSegment(1:end-1) > 0 & astSegment(2:end) <= 0);
        numWaves_ast = length(crossings_ast) - 1;

        if numWaves_ast > 50
            IG.zdcFlag_AST_hour(i) = 1;
        end

        for j = 1:numWaves_ast
            wave = astSegment(crossings_ast(j):crossings_ast(j+1));
            TS_AST_waveHeight(j) = max(wave) - min(wave);
            TS_AST_wavePeriod(j) = (crossings_ast(j+1) - crossings_ast(j)) / Fs;
        end

        IG.TS_AST_waveHeight(i) = max(TS_AST_waveHeight);
        IG.TS_AST_wavePeriod(i) = max(TS_AST_wavePeriod);
    
        % === Pressure ===
        crossings_press = find(pressSegment(1:end-1) > 0 & pressSegment(2:end) <= 0);
        numWaves_press = length(crossings_press) - 1;
    
        if numWaves_press > 50
            IG.zdcFlag_press_hour(i) = 1;
        end
    
        for j = 1:numWaves_press
            wave = pressSegment(crossings_press(j):crossings_press(j+1));
            TS_press_waveHeight(j) = max(wave) - min(wave);
            TS_press_wavePeriod(j) = (crossings_press(j+1) - crossings_press(j)) / Fs;
        end
 
            IG.TS_press_waveHeight(i) = max(TS_press_waveHeight);
            IG.TS_press_wavePeriod(i) = max(TS_press_wavePeriod);
       
end
IG.time_hourly = Raw.time(1:samplesInHour:end);
IG.time_hourly(end) = [];
end
    
%%
function Hm0_wavelet = run_wavelet_analysis(ast_signal, IG_band)
    fb = cwtfilterbank('SignalLength', length(ast_signal), 'SamplingFrequency', 1, 'Wavelet', 'amor', 'FrequencyLimits', IG_band);
    [cfs, freqs] = wt(fb, ast_signal);
    power = abs(cfs).^2;
    df = mean(diff(flip(freqs)));

    totalWaveletEnergy = sum(sum(power * df));
    totalSignalEnergy = sum(ast_signal .^ 2) / length(ast_signal);
    scaleFactor = totalSignalEnergy / totalWaveletEnergy;
    power = power * scaleFactor;

    goodFreq = (freqs >= IG_band(1)) & (freqs <= IG_band(2));
    m0_time = sum(power(goodFreq, :) * df, 1);
    Hm0_wavelet = 4 * sqrt(m0_time);
end