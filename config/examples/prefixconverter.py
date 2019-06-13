
def ConvertFromDownLoadToPrefix(prefix,options):
  import re
  import sys
  # delete arguments from first round of configure
  sys.argv = sys.argv[:1]
  options = [re.sub('download-([\w_]*)[=1]*$','with-\\1-dir='+prefix,arg) for arg in options]
  options = [arg for arg in options if not arg.startswith('--prefix') and not arg.startswith('--with-prefix-replace')]
  options = [arg for arg in options if not arg.startswith('--download-sowing-public')]
  return options

