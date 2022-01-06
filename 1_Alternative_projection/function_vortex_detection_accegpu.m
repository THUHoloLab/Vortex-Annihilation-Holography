function [num_po,num_ne]=function_vortex_detection_accegpu(pha,dh)

[n,m]=size(pha);
[xx,yy]=meshgrid(linspace(-m*dh/2,m*dh/2,m),linspace(-n*dh/2,n*dh/2,n));

% curl
pha_gy=exp(1i*diff(pha));
pha_gx=exp(1i*diff(pha,1,2));
gy=[angle(pha_gy);zeros(1,m)];
gx=[angle(pha_gx),zeros(n,1)];
gy_m1=[gy(:,2:m),zeros(n,1)];
gx_n1=[gx(2:n,:);zeros(1,m)];
g_curl=gather(gx+gy_m1-gx_n1-gy);

vor_po=im2double(imbinarize(g_curl, 2*pi-0.1));
vor_ne=im2double(imbinarize(-g_curl, 2*pi-0.1));
%{
figure
pcolor(x,y,g_curl);
shading interp;
colorbar;
colormap(hot);
figure
pcolor(x,y,vor_po);
shading interp;
colorbar;
colormap(hot);
figure
pcolor(x,y,vor_ne);
shading interp;
colorbar;
colormap(hot);
%}
% x_po=gpuArray(vor_po.*xx); y_po=gpuArray(vor_po.*yy);
x_po=vor_po.*xx; y_po=vor_po.*yy;
xv_po=x_po(find(x_po~=0))+dh/2; yv_po=y_po(find(y_po~=0))+dh/2;
num_po=length(xv_po);

x_ne=vor_ne.*xx; y_ne=vor_ne.*yy;
xv_ne=x_ne(find(x_ne~=0))+dh/2; yv_ne=y_ne(find(y_ne~=0))+dh/2;
num_ne=length(xv_ne);

end
