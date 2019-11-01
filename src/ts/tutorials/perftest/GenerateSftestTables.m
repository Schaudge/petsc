function GenerateSftestTables(results)
%  Generate performance tables for results obtained with ReadInSftestDirs()

  'tab:onenode-0'

  % First table communication from rank 0 within a node
  for i=1:size(results,2)
    sub = results(i).mat(2:end,1:end);
    if (results(i).maxpairinternode == 0) && (results(i).maxsingleinternode == 0) && (results(i).maxpairintersocket == 0) && (results(i).maxpairintrasocket == 0)  && (sum(sum(sub)) == 0)
      i
      round([full(results(i).maxsingleintrasocket),      full(results(i).maxsingleintersocket),      full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end

  'tab:onenode-0-nvshmem'

   % Second table communication from rank 0 within a node with nvshmem
  for i=1:size(results,2)
    sub = results(i).mat(2:end,1:end);
    if (results(i).maxpairinternode == 0) && (results(i).maxsingleinternode == 0) && (results(i).maxpairintersocket == 0) && (results(i).maxpairintrasocket == 0)  && (sum(sum(sub)) == 0)
      i
      round([full(results(i).maxsingleintrasocket),      full(results(i).maxsingleintersocket),      full(results(i).nvshmemStartup),      full(results(i).nvshmemBandwidth)],1)
    end
  end

  'tab:onenode-0-partner'

  % Third table communication from rank 0 within a node with partner also sending a message
  for i=1:size(results,2)
    sub = results(i).mat(2:end,2:end);
    if (results(i).maxpairinternode == 0) && (results(i).maxsingleinternode == 0) && (results(i).maxsingleintersocket == 0) && (results(i).maxsingleintrasocket == 0) && (sum(sum(sub)) == 0)
      i
      round([full(results(i).maxpairintrasocket),      full(results(i).maxpairintersocket),      full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end

  'tab:onenode-socket-0-all'
  
% Fourth table communication messages from socket  0 within a node with no partner  sending a message
  for i=1:size(results,2)
    sub = results(i).mat(4:end,:);
    if (results(i).maxpairinternode == 0) && (results(i).maxsingleinternode == 0) && (results(i).maxpairintersocket == 0) && (results(i).maxpairintrasocket == 0)  && (sum(sum(sub)) == 0)
      i
      round([full(results(i).maxsingleintrasocket),      full(results(i).maxsingleintersocket),      full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end

   'tab:onenode-socket-0-all-partner'
  
% Fifth table communication messages from socket  0 within a node with  partner  sending a message
  for i=1:size(results,2)
    if (results(i).maxpairinternode == 0) && (results(i).maxsingleinternode == 0) && (results(i).maxsingleintersocket == 0) && (results(i).maxsingleintrasocket == 0)
      i
      round([full(results(i).maxpairintrasocket),      full(results(i).maxpairintersocket),      full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end

  'tab:twonode-socket-0-all'

% Table with two nodes
  for i=1:size(results,2)
    if (size(results(i).mat,1) == 12) && (results(i).maxpairinternode == 0)  && (results(i).maxpairintersocket == 0) && (results(i).maxpairintrasocket == 0)
      i
      round([ full(results(i).maxsingleintrasocket),      full(results(i).maxsingleintersocket),  full(results(i).maxsingleinternode),     full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end

  'tab:twonode-socket-0-all-partner'

  % Table with two nodes partner exchange
  for i=1:size(results,2)
    if (size(results(i).mat,1) == 12) && (results(i).maxsingleinternode == 0)  && (results(i).maxsingleintersocket == 0) && (results(i).maxsingleintrasocket == 0)
      i
      round([ full(results(i).maxpairintrasocket),      full(results(i).maxpairintersocket),  full(results(i).maxpairinternode),     full(results(i).mpiStartup),      full(results(i).mpiBandwidth)],1)
    end
  end
