clear;
close all;
clc;
%% parameter
load('object_grayscale');
F1 =imresize(F1,[512,512]);[n,m]= size(F1);
E=sum(sum(F1));El=0.5*E;
lamda=532e-6;k=2*pi/lamda;
dh=0.00374;
F=abs(sqrt(F1));F=padarray(F,[n/4,m/4]);[nn,mm]=size(F);
phi=exp(1i*2*pi*rand(nn,mm));
amp=rand(nn,mm);
%% band-limitation
bandlim_spe=padarray(ones(nn/2,mm/2),[nn/4,mm/4]);
bandlim_in=ones(n,m);
bandlim_in=padarray(bandlim_in,[n/4,m/4]);
bandlim_ou=ones(nn,mm)-bandlim_in;
%% incident
w = 0.26;
[ox,oy] = meshgrid(linspace(-dh*mm/2 , dh*mm/2 , mm) , linspace(-dh*nn/2 , dh*nn/2 , nn));
Gaussian = exp((-1)*((ox.^2)+(oy.^2))./w);
incident=bandlim_spe.*Gaussian;
%% first iteration
loop=300;
RMSE=zeros(loop,1);NUM_PO=zeros(loop,1);NUM_NE=zeros(loop,1);
figure
for i=2:loop 
   amp=bandlim_in.*F+bandlim_ou.*amp;
   E1=amp.*phi;
   E2=fftshift(fft2(fftshift(E1)));
   E2_ave=sqrt((E+El)*incident.^2/sum(sum(incident.^2)));
   E2_k=E2_ave.*exp(1i*angle(E2));
   es=fftshift(ifft2(fftshift(E2_k)));
   amp=abs(es);
   amp_in=bandlim_in.*amp; amp_ou=bandlim_ou.*amp;
   amp=sqrt(E*(amp_in.^2)/sum(sum(amp_in.^2)))+sqrt(El*(amp_ou.^2)/sum(sum(amp_ou.^2)));
   I=amp((nn/2-n/2)+1:(nn/2+n/2),(mm/2-m/2)+1:(mm/2+m/2)).^2;
   I=E*I/sum(sum(I));
   P=mod(angle(es),2*pi);
   P=P((nn/2-n/2)+1:(nn/2+n/2),(mm/2-m/2)+1:(mm/2+m/2));
   imshow(I);
   Diff=double(I)-double(F1);
   MSE=gather(sum(Diff(:).^2)/numel(I));
   RMSE(i,1)=sqrt(MSE);
   diff_RMSE=RMSE(i,1)-RMSE(i-1,1);
   if abs(diff_RMSE)<0.00023 && RMSE(i,1)>0.035
      pha =angle(es);
      pha_in=pha.*bandlim_in;
      [pha_vfree]=function_vortex_elimination_accegpu(pha_in,dh);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(pha_vfree,dh);
      pha_vfree=bandlim_in.*pha_vfree+bandlim_ou.*pha;
      phi=exp(1i*pha_vfree);
   else
      phi=exp(1i*angle(es));
      phi_in=angle(phi).*bandlim_in;
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(phi_in,dh);
   end
end
phase=angle(phi);
An=angle(E2_k);
hologram=incident.*exp(1i*An);
%% output
Rec=fftshift(ifft2(fftshift(hologram)));
I=abs(Rec).^2;
I=I((nn/2-n/2)+1:(nn/2+n/2),(mm/2-m/2)+1:(mm/2+m/2));
I=E*(I/sum(sum(I)));
NUM=NUM_PO+NUM_NE;
figure,imshow(I);


