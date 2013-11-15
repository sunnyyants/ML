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
temp = zeros(size(theta));

for i = 1:m
    hypothesis = sigmoid(X(i,:) * theta);
       %t = (-y(i) * log(hypothesis) - (1-y(i)) * log((1-hypothesis)));
    J = J + (-y(i) * log(hypothesis) - (1-y(i)) * log((1-hypothesis)));
    for j = 1:size(theta,1)
        temp(j) = temp(j) + (hypothesis - y(i)) * X(i,j);
    end
end


J = J / m;
buffer = theta(2:end)'*theta(2:end);

J = J + (lambda / (2*m)) * buffer;

for i = 1:size(grad,1)
    if(i == 1)
        grad(i) = (1/m) * temp(i);
    else
        grad(i) = (1/m) * temp(i) + (lambda/m) * theta(i);
    end
end




% =============================================================

end
