function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

Z = size(X,1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%feed to hidden layer
Z = sigmoid(X * Theta1'); %a vector of sigmoided dotted product between X and theta1

% Add ones to the Z vector
Z = [ones(m, 1) Z];

%feed to output layer
for i=1:m
	tmp = size(num_labels,1);
	for c=1:num_labels
		tmp(c) = Z(i,:) * Theta2(c,:)';%dotted product for each row of X for each num_labels values
		%no need for sigmoid as we need only to see the max prob among 1 to num_labels
	endfor;
	[val idx]=max(tmp, [], 2);%search for the max prob index
	p(i)=idx;
endfor;





% =========================================================================


end
