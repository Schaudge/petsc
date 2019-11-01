function [time,timestack,labels] = PerformanceModelOneNode(A,nvshmem)
if (~exist('nvshmem', 'var'))
  nvshmem = false;
end

%
% each row represents the amount of data sent to the other ranks
%
%  Data from Jacob's first run on one node; e.g. the last rank receives from everyone else
rankdata = [ 0 0 0 0 0 0 ; ...
             335872 0 0 0 0 0 ; ...
             354176 467904 0 0 0 0 ; ...
             0 212672 284864 0 0 0 ; ...
             276736 0 212032 285120 0 0 ; ...
             258120 261640 53256 385160 503944  0];

rankdata = A - diag(diag(A));

% each row represents the amount of data passed from that socket to the other socket
intersockdata = [sum(sum(rankdata(1:3,4:6))) ; ...
                 sum(sum(rankdata(4:6,1:3)))];

% each row is the number of messages from each GPU to the other socket
% this formula is pure speculation
if (nvshmem == true)
  intersockcount = [max(rankdata(1:3,4:6)'>0) ; ...
                    max(rankdata(4:6,1:3)'>0)];
else
  intersockcount = [sum(rankdata(1:3,4:6)'>0) ; ...
                    sum(rankdata(4:6,1:3)'>0)];
end

% each row is the amount of data passed from a GPU to GPUs on the same socket
intrasockdata = [rankdata(1:3,1:3) ; ...
                 rankdata(4:6,4:6) ];

% each row is the number of messages from each GPU to the same socket
% this formula is pure speculation
if (nvshmem == true)
  intrasockcount = [max(rankdata(1:3,1:3)'>0) ; ...
                    max(rankdata(4:6,4:6)'>0)];
else
  intrasockcount = [sum(rankdata(1:3,1:3)'>0) ; ...
                    sum(rankdata(4:6,4:6)'>0)];
end

% numbers from Junchao's PetscSF ping-pong test
intrasockbandwidth = (1e-6)*47.2*(1024)^3;
intersockbandwidth = (1e-6)*35.8*(1024)^3;
if (nvshmem == true)
  % from table posted by Junchao in slack channel ecp-paper-feb-17, just the first number, not least squares
  intrasocklatency   = 33.4;
  intersocklatency   = 33.8;
else
  % from second report, computed by least squares
  intrasocklatency   = 23.3;
  intersocklatency   = 24.3;
end

if (nvshmem == true)
  kernellatency = 0;
else
  % from first report
  kernellatency = 10;
end

% Theoretical number based on GPU memory bandwidth and that each pack variable requires loading the index of the value,
%      the value, and then storing the result
% Number may be too high.
packbandwidth = (1e-6)*(900/3)*(1024)^3;

% assuming all communication channels can run at full speed at the same time
% also assumes everyone finishes packing before anything starts communicating (which is wrong)
% assume that each message comes with its own latency on that rank
%   For nvshmem there may be a single latency on that rank plus a smaller latency for each other rank it communicates with
time = 2*kernellatency + max(sum(rankdata)/packbandwidth + sum(rankdata')/packbandwidth) + ...
       max(max(intersocklatency*intersockcount'+intrasocklatency*intrasockcount')) + ...
       max( max(intersockdata'/intersockbandwidth), max(max(intrasockdata'/intrasockbandwidth)) );

%  split each max contribution into its two parts
[m,i] = max(sum(rankdata')/packbandwidth + sum(rankdata)/packbandwidth);
[m,j] = max(intersocklatency*intersockcount' + intrasocklatency*intrasockcount',[],'all','linear');
row = 1 + floor((j-1)/2);
col = 1 + mod(j-1,2);


timestack = full([2*kernellatency ; sum(rankdata(i,:))/packbandwidth ; sum(rankdata(:,i))/packbandwidth ; ...
             intrasocklatency*intrasockcount(row,col) ; intersocklatency*intersockcount(row,col) ;  ...
             max( max(intersockdata/intersockbandwidth), max(max(intrasockdata/intrasockbandwidth)) )]);

labels = flip([...
          'Communication bandwidth time        ';...
          'Inter-socket latencies              ';...
          'Intra-socket latencies              ';...
          'Unpack bandwidth time               ';...
          'Pack bandwidth time                 ';...
          'Pack/unpack kernel launch latencies ']);

% bug in Matlab, does not do stacked correctly for a single set of data
dummy = [timestack' ; 0 0 0 0 0 0];
barh(dummy,'stacked');
% It is impossible to get the colors of the legend in the same order as the bar graph!
legend(labels);
xlabel('Time');

display(time)

display(labels  + string([2; 1; 1; full(intrasockcount(row,col)); full(intersockcount(row,col)) ; 0]) + ['  ';'  ';'  ';'  ';'  ';'  '] + string(timestack))
