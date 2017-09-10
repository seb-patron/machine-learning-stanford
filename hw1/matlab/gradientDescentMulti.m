function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% X
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%     X = featureNormalize(X);
    theta_iters = size(theta);
    tempTheta = theta;
    
    % for each feature in theta, we iterate using j
    % at theta of j, we execute gradient descent, finding the new theta
    % values
    % once we found them all, we break this loop, and set actual theta to
    % these values. We continue for all iterations required
    for j = 1:theta_iters
        tempTheta(j, 1) = theta(j, 1) - alpha * (1/m) * sum( ((X * theta) - y) .* X(:, j));
        
    end


theta = tempTheta;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
