function [ps,pn]=function_envelope(N,M,pix)

% This function provides the envelope function of double phase encoding
a=pix;
[u,v]=meshgrid(linspace(-1/(2*pix),1/(2*pix),M),linspace(-1/(2*pix),1/(2*pix),N));
ps=cos(pi.*a.*u).*cos(pi.*a.*v).*sinc(a.*v).*sinc(a.*u);
pn=sin(pi.*a.*u).*sin(pi.*a.*v).*sinc(a.*v).*sinc(a.*u);
pn=abs(pn);
ps=abs(ps);
pn=1-pn/max(max(pn));
ps=ps/max(max(ps));
end