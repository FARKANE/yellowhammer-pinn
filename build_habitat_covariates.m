% build_habitat_covariates.m
% Covariate-only habitat predictor eta_hab for the PINN (review comment 2).
% Same construction as the GLM4 map in the benchmark script
% Matlab_src/LIAS_internal_report_production.m, WITHOUT the Year term and
% WITHOUT the Longitude/Latitude terms.  Run it in the folder that contains the
% benchmark explicative variables and Coef_GLM_final_model.txt.
%
% Output: H_covariates_2002_2024.mat with eta_hab (ny x nx x 23), Mat_lat, Mat_long
% Python equivalent: tools/build_habitat_covariates.py

data = readtable('Coef_GLM_final_model.txt', 'Delimiter', '\n', 'ReadVariableNames', false);
Coef = table2array(data);
ind_BIO = [1 2];
ind_CLC = [18 12 20 23 2 21 24 25 29];
load('Indice_CLIM_2002_2024.mat')            % Mat_lat, Mat_long
load('Indice_BIO_2002_2024.mat')
load('Portions_CLC2000.mat'); CLC_2000 = Portion_CLC;
load('Portions_CLC2006.mat'); CLC_2006 = Portion_CLC;
load('Portions_CLC2012.mat'); CLC_2012 = Portion_CLC;
load('Portions_CLC2018.mat'); CLC_2018 = Portion_CLC;

eta_hab = zeros(size(Mat_lat,1), size(Mat_lat,2), 23);   % no intercept (H is standardised)
% Longitude / Latitude terms (Coef(2), Coef(3)) deliberately NOT added
j = 4;
for i = 1:length(ind_BIO)
    eval(['eta_hab = eta_hab + Coef(j)*bio_', num2str(ind_BIO(i)), ';']); j = j + 1;
end
for i = 1:length(ind_BIO)
    eval(['eta_hab = eta_hab + Coef(j)*bio_', num2str(ind_BIO(i)), '.^2;']); j = j + 1;
end
for p = 1:2                                    % CLC, then CLC.^2
    for i = 1:length(ind_CLC)
        for ii = 1:23
            if ii < 5,      C = CLC_2000;      % 2002-2005
            elseif ii < 11, C = CLC_2006;      % 2006-2011
            elseif ii < 17, C = CLC_2012;      % 2012-2017
            else,           C = CLC_2018;      % 2018-2024
            end
            eta_hab(:,:,ii) = eta_hab(:,:,ii) + Coef(j) * C(:,:,ind_CLC(i)).^p;
        end
        j = j + 1;
    end
end
% Year term (Coef(j), j = 26) deliberately NOT added
save('H_covariates_2002_2024.mat', 'eta_hab', 'Mat_lat', 'Mat_long', '-v7')
