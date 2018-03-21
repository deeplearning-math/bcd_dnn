function val = relu_prox(a,b,gamma,d,N)
    val = zeros(d,N);
    x = (a+gamma*b)/(1+gamma);
    y = min(b,0);
	val((a+gamma*b >= 0 & b >=0) | (a*(gamma-sqrt(gamma*(gamma+1))) <= gamma*b & b < 0)) = x((a+gamma*b >= 0 & b >=0) | (a*(gamma-sqrt(gamma*(gamma+1))) <= gamma*b & b < 0));
    val(-a <= gamma*b & gamma*b <= a*(gamma-sqrt(gamma*(gamma+1)))) = b(-a <= gamma*b & gamma*b <= a*(gamma-sqrt(gamma*(gamma+1))));
	val(a+gamma*b < 0) = y(a+gamma*b < 0);
end