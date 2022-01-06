clear;
close all;
clc;
%% parameter
load('object_grayscale');
F1 =imresize(F1,[512,512]);[nn,mm]=size(F1);E=sum(sum(F1));
lamda=532e-6;k=2*pi/lamda;
dh=0.00374;
F=abs(sqrt(F1));
phi=exp(1i*2*pi*rand(nn,mm));
%% band-limitation
bandlim_spe=padarray(ones(nn/2,mm/2),[nn/4,mm/4]);
%% alternative projection
loop=100;
RMSE=zeros(loop,1);NUM_PO=zeros(loop,1);NUM_NE=zeros(loop,1);
figure
for i=2:loop 
   amp=F;
   E1=amp.*phi;
   E2=fftshift(fft2(fftshift(E1)));
   E2_ave=sqrt(E*((abs(E2).*bandlim_spe).^2)/sum(sum((abs(E2).*bandlim_spe).^2)));
   E2_k=E2_ave.*exp(1i*angle(E2));
   es=fftshift(ifft2(fftshift(E2_k)));
   amp=abs(es);
   amp=sqrt(E*(amp.^2)/sum(sum(amp.^2)));
   I=amp.^2;
   I=E*I/sum(sum(I));
   P=mod(angle(es),2*pi);
   imshow(I);
   Diff=double(I)-double(F1);
   MSE=gather(sum(Diff(:).^2)/numel(I));
   RMSE(i,1)=sqrt(MSE);
   diff_RMSE=RMSE(i,1)-RMSE(i-1,1);
   if abs(diff_RMSE)<0.0005 && RMSE(i,1)>0.03
      pha =angle(es);
      [pha_vfree]=function_vortex_elimination_accegpu(pha,dh);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(pha_vfree,dh);
      phi=exp(1i*pha_vfree);
   else
      phi=exp(1i*angle(es));
      phi_in=angle(es);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(phi_in,dh);
   end
end
phase=angle(phi);
An=angle(E2_k);
Am=abs(E2_k);
hologram=Am.*exp(1i*An);
%% Reconstruction
e=fftshift(ifft2(fftshift(hologram)));
I=abs(e).^2;
I=E*(I/sum(sum(I)));
NUM=NUM_PO+NUM_NE;
figure,imshow(I);