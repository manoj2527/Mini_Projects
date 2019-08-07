function [outH, outIdx] = myRansac(x1,x2,fitFcn,distFcn,thresh,iter)
    
    n = size(x1,1);
    len = 0;
    outH = 0;
    for i=1:iter
        Idx = randperm(n,4);
        x11 = x1(Idx,:);
        x22 = x2(Idx,:);
       
        H = fitFcn(x11,x22);
        D = distFcn(H,x1,x2);
        
        inlier = find(D < thresh);
        if(len < size(inlier,2))
            outH = H;
            outIdx = Idx;
            len = size(inlier,2);
        end
    end
end