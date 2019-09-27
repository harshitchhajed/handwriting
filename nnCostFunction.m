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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Adds a column of 1's to X
X = [ones(m, 1) X];

% Using Feed forwrad propagation to predict the hypothesis with the given
% theta values. finding a1, a2, and a3 for neural network a3 = hypothesis
a1 = X';

z2 = Theta1 * a1;
a2 = sigmoid(z2);

% bias unit for second layer
a2 = [ones(1, m); a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

h = a3;

% The following for-loop changes the y-values from showing the digit to a
% matrix with 1 at the index of digit and 0 other wise. For eg. 5 is
% changed to [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
A = y';
new_y = zeros(num_labels, m);

for i = 1:m
    y_val = A(i);
    new_y(y_val, i) = 1;
end

% Unregularized part of cost function of neural network. Note that it uses
% new_y
intermediate1_J = (1/m) * (-new_y .* log(h) - (1-new_y) .* log(1-h));

% Regularized part of cost function
temp1 = Theta1;
temp2 = Theta2;

% Set the first column of Theta (or temp) to zero because it corresponds to
% the bias unit
temp1(:, 1) = 0;
temp2(:, 1) = 0;

% Handling the squaring part separately. Note temp1' * temp1 is not the
% same as temp1.^2
A = temp1.^2;
B = temp2.^2;

% Regularized Part of J
intermediate2_J = (lambda/(2*m)) * sum(A, 'all') + (lambda/(2*m)) * sum(B, 'all');

% Add both the Js
J = sum(intermediate1_J, 'all') + intermediate2_J;


%GRADIENTS
% Follow the back propagation algorithm. 

% Set the accumalators for each Theta gradient
capital_delta1 = 0;
capital_delta2 = 0;

% Calculate the error in a3 and a2. There is no error in a1 because that is
% the data
delta3 = a3 - new_y;

% Create a new theta without the learning parameters for the bias unit
% because that is always 1 and has no error.
new_Theta2 = Theta2(:, 2:end);

% This is just the formula to compute all the deltas backward
delta2 = (new_Theta2' * delta3) .* sigmoidGradient(z2);

% Again, just the formula to get the Theta_grads
capital_delta1 = capital_delta1 + delta2 * a1';
capital_delta2 = capital_delta2 + delta3 * a2';


Theta1_grad(:, 1) = (1/m) * capital_delta1(:, 1);
Theta2_grad(:, 1) = (1/m) * capital_delta2(:, 1);

Theta1_grad(:, 2:end) = (1/m) * capital_delta1(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = (1/m) * capital_delta2(:, 2:end) + (lambda/m) * Theta2(:, 2:end);


% Unroll gradients because function minimization routine only accepts
% vectors as Theta and Theta_grad

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
