function [RankMat] = GRISILI(X, R, L_array, alpha_min)
%GRISILI Python-facing adapter for upstream GRISLI.
% X is a cells-by-(1+genes) spacetime matrix: first column is pseudotime,
% remaining columns are gene expression values.

if nargin < 2 || isempty(R)
    R = 100;
end
if nargin < 3 || isempty(L_array)
    L_array = 10:10:size(X, 2);
end
if nargin < 4 || isempty(alpha_min)
    alpha_min = 0.3;
end

[~, I] = sort(X(:, 1));
X = X(I, :);

Alpha = @(Kx, Dt, sigx, sigt) ...
    exp(-(Kx.^2) / (2 * sigx^2)) .* ...
    exp(-(Dt.^2) / (2 * sigt^2)) .* ...
    (Dt.^2);

Rnk_array_TIGRESS_area_L = Compute_A_app_wo_ref(X, L_array, Alpha, R, alpha_min);
RankMat = mean(Rnk_array_TIGRESS_area_L, 3);

end
