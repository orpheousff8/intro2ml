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

%cost function

h = sigmoid(X * theta);%vector matrix

%cost function
J = sum(-y.*log(h) - (1-y).*log(1-h))/m;

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

% =============================================================

end
