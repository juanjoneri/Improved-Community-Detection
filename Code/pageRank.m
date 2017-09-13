function  H  = pageRank( F, W, alpha, diffStep )

    n = size(W,1);
    v = sum(W,2);
    v = 1./v;
    Dinv = spdiags(v,0,n,n);
    M = alpha * W * Dinv;
    Fnormalized = bsxfun(@rdivide, F, sum(F));

    H = Fnormalized; % dividido por el vertice de cada uno Po

    for i = 1:diffStep
        H = M*H + (1-alpha)*Fnormalized; %% NORMALIZE F
    end

end


