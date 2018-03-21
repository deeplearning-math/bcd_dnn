function Vstar = updateV_js(U1,U2,W,b,rho,gamma,act)
    [~,d] = size(W);
    I = sparse(eye(d));
    switch act
        case 1 
            U1 = max(0,U1); % ReLU
        case 2
            U1 = tanh_proj(U1); % tanh
        case 3
            U1 = sigmoid_proj(U1); % sigmoid
    end
%     alpha = norm(eta2*(M2'*M2)+eta1*eye(d));
    Vstar = (rho*(W'*W)+gamma*I)\(rho*W'*(U2-b)+gamma*U1);

end