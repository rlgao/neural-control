clc;clear;close all;

nsteps = 100;
for k = 1:nsteps
    y(k+1) = (y(k)-y(k-1)) / sqrt(1+y(k)*y(k)) + u(k)*u(k)*u(k);
end
