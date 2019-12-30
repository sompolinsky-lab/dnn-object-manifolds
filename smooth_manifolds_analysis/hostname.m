function name=hostname()
% Get the hostname in a persistent, consistent way.
persistent hostnamePersistent

if isempty(hostnamePersistent)
   [ret, name] = system('hostname');
   if ret==0
      hostnamePersistent = name;
   end
end
name = hostnamePersistent;
