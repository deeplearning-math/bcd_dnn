function [Wstar,bstar] = updateWb_js(U,V,W,b,alpha,rho)
    [d,N] = size(V);
    I = sparse(eye(d));
    Wstar = (alpha*W+rho*(U-b)*V')/(alpha*I+rho*(V*V'));
    bstar = (alpha*b+rho*sum(U-W*V,2))/(rho*N+alpha);
end