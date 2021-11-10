function Ap = buildmatrix(ns,nn,np)
%
%  Build communication matrix for ns subdomains with a maximum of nn neighbors for each
%
A = zeros(ns,ns);
for i=1:ns-1
  s = nn - sum(A(i,:) ~= 0);
  r = randi([i+1 ns],[1 s]);
  A(i,r) = randi([0 100],[1 s]);
  A(r,i) = randi([0 100],[1 s]);
end
%
%  Replace each node in the graph with np nodes each likely fully connected with various weights
%
Ap = zeros(ns*np,ns*np);
for i=1:ns
  for j=1:ns
    Ap(np*(i-1)+(1:np),np*(j-1)+(1:np)) = round(rand(np)*A(i,j));
  end
end

