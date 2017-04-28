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

%cost function
h = X * theta;%vector matrix

%cost function
J = sum((realpow(h - y,2))/ (2 * m));

%regularisation (skip theta0)
R = (theta'*theta - realpow(theta(1),2)) * lambda / (2 * m);

J += R;

%gradient
for j=1:size(theta,1)
	grad(j) = (h-y)' * X(:,j)/m; %dotted product for each column of X
endfor;
%regularisation to all theta
grad = grad .+ lambda * theta / m;
%remove regularisation for theta0
grad(1) -= lambda * theta(1) / m;










% =========================================================================

grad = grad(:);

end
