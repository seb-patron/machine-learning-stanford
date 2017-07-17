function [ J, grad ] = test(a)
% This is a script for comparing the output from cofiCostFunc.m
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 0;
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

% output:
% J =  311.63
% 
% grad =
%   -16.1880
%   -23.5440
%    -5.1590
%   -14.9720
%   -21.4380
%   -30.4620
%    -6.5660
%   -19.5440
%    -3.4230
%    -7.0280
%    -3.4140
%   -12.2590
%   -16.0600
%    -9.7420


end

