function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

valuesToTest = [0.01 0.03 0.1 0.3 1 3 10 30];
results = zeros(64,3);
resIndex = 1;

for i = 1:length(valuesToTest)
  for j = 1:length(valuesToTest)
    model= svmTrain(X, y, valuesToTest(i), @(x1, x2) gaussianKernel(x1, x2, valuesToTest(j)));
    predictions  =  svmPredict(model, Xval);
    results(resIndex, 1) = mean(double(predictions ~= yval));
    results(resIndex, 2) = valuesToTest(i);
    results(resIndex, 3) = valuesToTest(j);
    resIndex = resIndex + 1;
  end
end

[minVal index] = min(results(:,1));
C = results(index, 2);
sigma = results(index, 3);





% =========================================================================

end
