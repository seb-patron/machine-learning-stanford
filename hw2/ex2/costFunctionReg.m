function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


tempTheta = theta;

% Previous code for cost function
J = (-1/m) * sum( ((1 .* y) .* log(sigmoid(X*tempTheta))) + ((1-y) .* log(1 - sigmoid(X*tempTheta))) );

% now we regularize, from theta 1 (skipping theta 0) and onwards
J = J + ( lambda/(2*m) ) * sum(theta(2:end,1) .^2);
iters = size(theta(:,1));

% iters = m
% Don't ruglarize theta 0, do normal gradient find
grad(1,1) = (1/m) * sum((sigmoid(X*tempTheta) - y) .* X(:,1));

if (iters(:,1) > 1)
    for j = 2:iters
        grad(j,1) = (1/m) * sum((sigmoid(X*tempTheta) - y) .* X(:,j)) + ((lambda/m) * theta(j,1));
    end
end




% =============================================================

end
