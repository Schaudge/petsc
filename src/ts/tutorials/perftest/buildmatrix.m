function A = buildmatrix(ns,nn,np)
%
%  Build communication matrix for ns ranks with a maximum of nn neighbors for each
%
%  Note: some rows will have more than ns connections due to enforcing the symmetry
%
A = zeros(ns,ns);
for i=1:ns-1
  % could select from a normal distributions the number of neighbors for the ith rank instead of always using nn
  %
  % s counts the number of additional connections needed for rank i to have nn total connections
  s = min(nn - sum(A(i,:) ~= 0),ns-i);
  if s > 0
    r = i + randperm(ns - i,s);
    % ensure the communication pattern has a symmetric non-zero structure
    A(i,r) = randi([0 100],[1 s]);
    A(r,i) = randi([0 100],[1 s]);
  end
end


