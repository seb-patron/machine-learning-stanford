function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% y
tempX = [ones(m, 1) X];
size(tempX)
size(Theta1)
% h_of_x1 = sigmoid(tempX * Theta1');
% h_of_x1 = [ones(m, 1) X];
% size(h_of_x1)
j = 0;
h_of_xK = zeros(m, num_labels);

% finds all possible output values and puts in matrix h_of_xK
% each row corresponds to an input, and each column is the guess of
% what the output is
for i=1:m
    a1 = sigmoid(tempX(i,:) * Theta1');
    a1 = [ones(1, 1) a1];

    a2 = sigmoid(a1 * Theta2');
    h_of_xK(i,:) = a2;

end

for i=1:m
     j = 0;
         for k=1:num_labels

            if y(i) == k
                yVal = 1;
            else
                yVal = 0;
            end
    %     h_of_xK(1,k)
    %     (-1 * yVal * log(h_of_xK(1,k))) 
            j = j + ( (-yVal * log(h_of_xK(i,k))) - ((1-yVal) * log(1 - h_of_xK(i,k))) );
         end
% end
     J = J + j;
     j = 0;
     
end
J = 1/m * J;



% NOW ADD REGULARIZATION
sum1 = 0;
theta1Rows = size(Theta1, 1);
theta1Cols = size(Theta1, 2);
for j=1:theta1Rows
    
    for k=2:theta1Cols
        sum1 = sum1 + (Theta1(j,k)^2);
    end
end

% set vars for function to work, including getting dimensions
% to use as number of iterations
sum2 = 0;
theta2Rows = size(Theta2, 1);
theta2Cols = size(Theta2, 2);
for j=1:theta2Rows
    
    for k=2:theta2Cols
        sum2 = sum2 + (Theta2(j,k)^2);
    end
end
regularization = lambda / (2 * m) * (sum1 + sum2);
J = J + regularization;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


for i=1:m
    % ------ NOTE!! ------------
    % a2 is actually renamed from a1, a3 is renamed from a2
    % in future code can be refactored to us prev found values in
    % above equation
    
    % input layer l=1
    a1 = X(i,:)';
    a1 = [1 ; a1];
    
    % hidden layer l=2
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    
    % hidden layer l=3
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Calculates delta 3 values
    delta3 = zeros(1, num_labels);
    for k=1:num_labels
        if y(i) == k
                yVal = 1;
            else
                yVal = 0;
        end
        delta3(k) = a3(k) - yVal;
            
    end
    delta3 = delta3';

    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    % no delta1 bc input has no error
    
    % remove bias units from delta2
    delta2 = delta2(2:end);
    
    % Big delta (gradient) update
    Theta1_grad = Theta1_grad + delta2 * a1'; % aka D1
    Theta2_grad = Theta2_grad + delta3 * a2'; % aka D2


end

% Regularization
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% fprintf('size of a1')













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
