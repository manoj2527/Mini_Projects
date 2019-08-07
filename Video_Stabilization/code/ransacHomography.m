function [ H, outIdx ] = ransacHomography( x1, x2, thresh )
%RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    
    iter = 500;
    [H, outIdx] = myRansac(x1,x2,@fitFcn,@distFcn,thresh,iter);
end

function H = fitFcn(x1,x2)
    H = homography(x1,x2);
end

function dist = distFcn(H,x1,x2)
    n = size(x1,1);
    x3 = H*[x1 ones(n,1)]';
    x3 = x3(1:2,:)./x3(3,:);
    dist = sqrt(sum((x2'-x3).^2));
end