clear;
close all;
clc;
%% parameter
tic;
load('object');
n=2160;m=3840;
lamda=[450e-6;450e-6;450e-6];
k=2*pi./lamda;
r=im2double(F1(:,:,1));g=im2double(F1(:,:,2));b=im2double(F1(:,:,3));
r=imresize(r,[n,m]);g=imresize(g,[n,m]);b=imresize(b,[n,m]);
rgb=cat(3,r,g,b);
D1=im2double(D1);
figure,imshow(D1);
D1=imresize(D1,[n,m]);
D1=padarray(D1,[n/2,m/2]);
D1=ceil(D1*255);
dh=0.00374;
Fr=abs(sqrt(r));Fg=abs(sqrt(g));Fb=abs(sqrt(b));
Fr=padarray(Fr,[n/2,m/2]);Fg=padarray(Fg,[n/2,m/2]);Fb=padarray(Fb,[n/2,m/2]);
Er=sum(sum(r));Eg=sum(sum(g));Eb=sum(sum(b));
EE=[Er;Eg;Eb];
F=cat(3,Fr,Fg,Fb);
Es=1.5;
figure,imshow(F);
[nn,mm]=size(Fr);
%% band-limitation
bandlim_spe=padarray(ones(nn/2,mm/2),[nn/4,mm/4]);
bandlim_in=bandlim_spe;
bandlim_ou=ones(nn,mm)-bandlim_in;
incident=bandlim_spe;
%% first iteration
ph_random=exp(1i*2*pi*rand(nn,mm));
inner_loop=400;
[fx,fy]=meshgrid(linspace(-1/(2*dh),1/(2*dh),mm),linspace(-1/(2*dh),1/(2*dh),nn));
MSE=zeros(inner_loop,1,3);
RMSE=zeros(inner_loop,1,3);
phase=zeros(nn,mm,3);
H=zeros(nn/2,mm/2,3);
An=zeros(nn/2,mm/2,3);
%% layer
slice=2;
step=256/slice;
figure
for j=1:3
% j=2;
Sm=m*dh;Sn=n*dh;
delta_m=(2*Sm).^(-1);delta_n=(2*Sn).^(-1);
object=zeros(nn,mm,slice);
E=zeros(1,slice);
El=zeros(1,slice);
Amp=zeros(nn,mm,slice);
fil=zeros(nn,mm,slice);
H_AS=zeros(nn,mm,slice);
h_AS=zeros(nn,mm,slice);
for s=1:slice
        [x,y]=find(D1>(slice-s)*step & D1<=(slice-s+1)*step);
    for jj=1:length(x)
        object(x(jj),y(jj),s)=F(x(jj),y(jj),j);
    end
Ir=object((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4),s).^2;
Amp(:,:,s)=bandlim_in.*object(:,:,s)+bandlim_ou.*rand(nn,mm);
        filter=fftshift(fft2(fftshift(object(:,:,s))));
        filter=log(1+abs(filter));
        fil(:,:,s)=(exp(filter/max(max(filter)))).^0.9;
        oz=150+50*(s-1)/(slice-1);
        lim_m=((2*delta_m*oz).^2+1).^(-1/2)./lamda(j,1);
        lim_n=((2*delta_n*oz).^2+1).^(-1/2)./lamda(j,1);
        bandlim_m=(lim_m-abs(fx));
        bandlim_n=(lim_n-abs(fy));
        bandlim_m=imbinarize(bandlim_m,0);
        bandlim_n=imbinarize(bandlim_n,0);
        bandlim_AS=bandlim_m.*bandlim_n;
        H_AS(:,:,s)=fil(:,:,s).*exp(1i*k(j,1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2));
        h_AS(:,:,s)=fil(:,:,s).*exp(1i*k(j,1)*(-1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2));
E(1,s)=sum(sum(Ir));
El(1,s)=3*E(1,s);
end
    amp=Amp;
    phi=exp(1i*2*pi*rand(nn,mm,slice));
for i=2:inner_loop
    Hl=zeros(nn,mm);
   for s=1:slice         
        E1=amp(:,:,s).*phi(:,:,s);
        E2=fftshift(fft2(fftshift(E1)));
        E2=ifftshift(ifft2(ifftshift(E2.*H_AS(:,:,s))));
        Hl=Hl+E2;
   end
   H2_k=sqrt(4*EE(j,1).*incident).*exp(1i*angle(Hl));
   es3=zeros(nn,mm,slice);
   I=zeros(nn/2,mm/2,slice);
   for s=1:slice
        es=fftshift(fft2(fftshift(H2_k)));
        es=ifftshift(ifft2(ifftshift(es.*h_AS(:,:,s))));
        es3(:,:,s)=es;
        I_amp=abs(es);
        amp_in=bandlim_in.*object(:,:,s);
        amp_ou=bandlim_ou.*abs(es);
        amp(:,:,s)=sqrt(E(1,s)*(amp_in.^2)/sum(sum(amp_in.^2)))+sqrt(El(1,s)*(amp_ou.^2)/sum(sum(amp_ou.^2)));
        I_amp=I_amp((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4)).^2;
        I_amp=EE(j,1)*I_amp/sum(sum(I_amp));
        I(:,:,s)=I_amp;
   end
   imshow(I(:,:,1));
   Diff=(imbinarize(Ir).*double(I))-double(Ir);
   MSE(i,1,j)=gather(sum(Diff(:).^2)/numel(I));
   RMSE(i,1,j)=sqrt(MSE(i,1,j));
   diff_RMSE=RMSE(i,1,j)-RMSE(i-1,1,j);
   if abs(diff_RMSE)<0.00005 && RMSE(i,1,j)>0.01
      pha =angle(object.*es3);
      for s=1:slice
      pha_in=gpuArray(pha(:,:,s).*bandlim_in);
      [pha_vfree]=function_vortex_elimination_accegpu(pha_in,dh);
      pha_vfree=gather(pha_vfree);
      pha_vfree=bandlim_in.*pha_vfree+bandlim_ou.*angle(es3(:,:,s));
      phi(:,:,s)=exp(1i*pha_vfree);
      end
   else
      phi=exp(1i*angle(es3));
   end
end
H(:,:,j)=Hl((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4));
An(:,:,j)=mat2gray(angle(H(:,:,j)));
imwrite(dph,['phase_type_',num2str(j),'hologram.bmp']);
end
%% output
I_rec=zeros(nn/2,mm/2,3);
oz=150;
for j=1:3
hologram=exp(1i*2*pi*An(:,:,j));
hologram=padarray(hologram,[nn/4,mm/4]);
e=fftshift(fft2(fftshift(hologram)));
h_AS=exp(1i*k(j,1)*(-1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2)); 
e=ifftshift(ifft2(ifftshift(e.*h_AS)));
rec=abs(e).^2;
rec=rec((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4));
I_rec(:,:,j)=EE(j,1)*(rec/sum(sum(rec)));
end
%% 
figure,imshow(I_rec);
toc;

