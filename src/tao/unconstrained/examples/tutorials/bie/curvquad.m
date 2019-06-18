function G = curvquad(G,rule,N,p)
% CURVQUAD - set up underlying quadrature for a closed curve struct
%
% G = curvquad(G,'ptr',N) adds N-node periodic trapezoid rule
% G = curvquad(G,'panel',N,p) adds panel rule with uniform-sized p-node panels
%  for at least N nodes total.
% Barnett 6/7/14

if strcmp(rule,'ptr')
  s = 2*pi*(1:N)'/N;
  G.w = 2*pi/N*ones(N,1);
  G.n = 1;                            % # panels
elseif strcmp(rule,'panel')
  G.p = p; n = ceil(N/p); G.n = n;                    % # panels
  [s1 w1] = gauss(p); s1 = (s1+1)/2; w1 = w1/2; % panel quad for [0,1]
  se = 2*pi*(0:n)/n;        % panel breaks (including first and last end)
  G.w = repmat(w1*2*pi/n, [n 1]);  % stack copies of weights together
  s = nan(n*p,1); for i=1:n, s((i-1)*p + (1:p)) = se(i)+(se(i+1)-se(i))*s1; end
end
% set up stuff indep of the quadr scheme...
G.s = s; G.x = G.Z(s); 
G.sp = abs(G.Zp(s)); G.nx = -1i*G.Zp(s)./G.sp;
G.cur = -real(conj(G.Zpp(s)).*G.nx)./G.sp.^2;  % curvatures

