 function [ H ] = homography( p1, p2 )
%HOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    I = size(p1,1);
    f1=p1;
    f2=p2;
    sumx1 = 0;
    sumy1 = 0;
    sumx2 = 0;
    sumy2 = 0;
    for i=1:I
        sumx1 = sumx1+p1(i,1);
        sumx2 = sumx2+p2(i,1);
        sumy1 = sumy1+p1(i,2);
        sumy2 = sumy2+p2(i,2);
    end
    meanx1 = sumx1/I;
    meanx2 = sumx2/I;
    meany1 = sumy1/I;
    meany2 = sumy2/I;
    c = zeros(2,2);
    for i=1:I
        p1(i,1) = p1(i,1)-meanx1;
        p1(i,2) = p1(i,2)-meany1;
        p2(i,1) = p2(i,1)-meanx2;
        p2(i,2) = p2(i,2)-meany2;
        c = c+p1(i,:)'*p2(i,:);
    end
    c = c/I;
    
    [U,~,V] = svd(c);
    W = [1,0;0,det(V*U')];
    R = V*W*U';
    R = R/det(R);
    T = zeros(2,1);
    for i=1:I
        T = T+(f2(i,:)'-R*f1(i,:)');
    end
    T = T/I;
    M = zeros(3,3);
    M(1:2,1:2) = R;
    M(1:2,3) = T;
    M(3,3)=1;
    H=M;
end

