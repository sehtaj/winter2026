function theta = l1ls(X,y,lambda)

%%% YOUR CODE HERE
[~, n] = size(X);
theta = zeros(n, 1);

% Maintain X * theta - y so each coordinate update is O(m).
residual = -y;
column_norm_sq = sum(X .^ 2, 1)';
tol = 1e-5;

while true
    theta_prev = theta;

    for i = 1:n
        if column_norm_sq(i) == 0
            theta(i) = 0;
            continue;
        end

        rho = -X(:, i)' * residual + column_norm_sq(i) * theta(i);
        new_theta_i = sign(rho) * max(abs(rho) - lambda, 0) / column_norm_sq(i);

        if new_theta_i ~= theta(i)
            residual = residual + X(:, i) * (new_theta_i - theta(i));
            theta(i) = new_theta_i;
        end
    end

    if norm(theta - theta_prev, inf) < tol
        break;
    end
end
