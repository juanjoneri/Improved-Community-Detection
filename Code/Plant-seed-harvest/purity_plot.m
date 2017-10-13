function [pur] = purity_plot( A,nc,maxiter,Cgt )
%A=0.5*(A+A')
%issymetric(A)==0, error
tstart=tic;
n=size(A,1);
v = sum(A,2);
v = 1./v;
Dinv = spdiags(v,0,n,n);
M = A * Dinv;
C=randi(nc,n,1);
ds=5*10^(-4)*n/nc;
s=1;
pur=zeros(1,maxiter);
for i=1:maxiter
    P=plant_seed(C,nc,floor(s));
    while min(min(P))<10^(-8)
        P = M*P;
    end
    
    [~,C] = max(P,[],2);
    pur(i)=purity(C, Cgt , nc);
    s=s+ds;
    
    if mod(i,10)== 0
        fprintf('At iteration %i \t time %.1f  purity=%.2f\n',i,toc(tstart),pur(i));
    end
end

plot(pur)
end