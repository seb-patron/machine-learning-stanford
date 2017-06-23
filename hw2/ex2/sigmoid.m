function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
num_rows = length((z(:,1))); % number of training examples
num_cols = length((z(1,:)));
% X
for row = 1:num_rows
    for col = 1:num_cols

        g(row,col) = 1 / (1 + exp(-1*z(row,col)));
        
    end
end



% =============================================================

end
