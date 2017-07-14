function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% fprintf('printing size of x, theta, and y')
% size(X)
% size(theta)
% size(y)
% size(grad)
J = 1/(2*m) * sum( ((X*theta)-y).^2 ); % Cost function
J = J + (lambda/(2*m) * sum( theta(2:end,1).^2)); % regularization


%first calculate hypothesis
h = X*theta;
error = h - y; % find error
gradChange = (X' * error) / m; %calculate new theta values (multiplication automatically sums up cols)
grad  = gradChange;
grad(2:end,1) = grad(2:end,1) + lambda/m *theta(2:end,1); %apply regularization (skip bias unit)



% =========================================================================

grad = grad(:);

end
