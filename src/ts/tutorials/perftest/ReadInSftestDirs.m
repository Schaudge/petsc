function results = ReadInSftestDirs(dir)
%
%  Reads in multiple files produced by sftest.c, see ReadInSftest.m
%
  files = ls([ dir '/*/OPT*/sftest_out*']);
  files = split(files);
  n = size(files,1);
  results = [];
  for f=1:n
    file = files{f};
    if endsWith(file,'.info')
      continue
    end
    result = ReadInSftest(file);
     % reject if sizes does not start with 0
     % reject if matrix sizes are 1 (mistake run)
    if size(result(1).mat,1) > 1 && (result(1).sizes(1) == 0) && (sum(sum(result(1).mat == 1))  ==  0)
      results = [results result];
    end
  end
