function y = lwlr(X_train, y_train, x, tau)

% YOUR ANSWERS HERE

[m, n] = size(X_train);
lambda = 0.0001;

w = zeros(m,1);

for i = 1:m
    xi = X_train(i,:)';
    difference = x - xi;
    distance_squared = difference' * difference;
    w(i) = exp(-distance_squared / (2 * tau * tau));
end

theta = zeros(n,1);

for iter = 1:10
    
    h = 1 ./ (1 + exp(-X_train * theta));
    
    z = zeros(m,1);
    for i = 1:m
        z(i) = w(i) * (y_train(i) - h(i));
    end
    
    grad = X_train' * z - lambda * theta;
    
    D = zeros(m,1);
    for i = 1:m
        D(i) = -w(i) * h(i) * (1 - h(i));
    end
    
    H = X_train' * diag(D) * X_train - lambda * eye(n);
    
    theta = theta - H \ grad;
    
end

prob = 1 / (1 + exp(-x' * theta));

if prob > 0.5
    y = 1;
else
    y = 0;
end

end