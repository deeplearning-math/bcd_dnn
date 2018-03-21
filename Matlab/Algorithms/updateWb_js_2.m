function [Wstar,bstar] = updateWb_js_2(U,V,W,b)
    [d,N] = size(V);
%     I = sparse(eye(d));
    Wstar = (U-b)*V'*pinv(V*V');
    bstar = sum(U-W*V,2)/N;
end