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
figure,imshow(F);
[nn,mm]=size(Fr);
[M1,M2]=function_binary_grating(nn,mm);
%% band-limitation
bandlim_spe=padarray(ones(nn/2,mm/2),[nn/4,mm/4]);
bandlim_in=bandlim_spe;
bandlim_ou=ones(nn,mm)-bandlim_in;
incident=bandlim_spe;
%% first iteration
ph_random=exp(1i*2*pi*rand(nn,mm));
inner_loop=100;
[fx,fy]=meshgrid(linspace(-1/(2*dh),1/(2*dh),mm),linspace(-1/(2*dh),1/(2*dh),nn));
[ps,pn]=function_envelope(nn,mm,dh);
MSE=zeros(inner_loop,1,3);
RMSE=zeros(inner_loop,1,3);
phase=zeros(nn,mm,3);
DPH=zeros(nn/2,mm/2,3);
%% layer
slice=2;
step=260/slice;
object=zeros(nn,mm,3);
fil=zeros(nn,mm,3);
figure
for j=1:3
Hl=zeros(nn,mm);
for s=1:slice
    [x,y]=find(D1>=(s-1)*step & D1<s*step);
    for jj=1:length(x)
        object(x(jj),y(jj),s)=F(x(jj),y(jj),j);
    end
Amp=object(:,:,s);
        filter=fftshift(fft2(fftshift(object(:,:,s))));
        filter=log(1+abs(filter));
        fil(:,:,s)=(exp(filter/max(max(filter)))).^0;
imshow(Amp);
Ir=Amp.^2;
E=sum(sum(Ir));
oz=150+30*(slice-s)/(slice-1);
   H_AS=ps.*pn.*exp(1i*k(j,1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2)); 
   h_AS=ps.*pn.*exp(1i*k(j,1)*(-1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2));
   phi=exp(1i*2*pi*rand(nn,mm));
for i=2:inner_loop 
   E1=Amp.*phi;
   E2=fftshift(fft2(fftshift(E1)));
   E2=ifftshift(ifft2(ifftshift(E2.*H_AS)));
   E2_am=incident.*(abs(E2));
   E2_am=sqrt(E*(E2_am.^2)./sum(sum(E2_am.^2)));
   E2_k=E2_am.*exp(1i*angle(E2));
   es=fftshift(fft2(fftshift(E2_k)));
   es=ifftshift(ifft2(ifftshift(es.*h_AS)));
   amp=abs(es);
   amp=sqrt(E*(amp.^2)./sum(sum(amp.^2)));
   I=amp((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4)).^2;
   imshow(mat2gray(I));
   Diff=double(I)-double(rgb(:,:,j));
   MSE(i,1,j)=gather(sum(Diff(:).^2)/numel(I));
   RMSE(i,1,j)=sqrt(MSE(i,1,j));
   diff_RMSE=RMSE(i,1,j)-RMSE(i-1,1,j);
   if abs(diff_RMSE)<0.0005 && RMSE(i,1,j)>0.005
     pha =angle(Amp.*es);
     pha_in=gpuArray(pha.*bandlim_in);
     [pha_vfree]=function_vortex_elimination_accegpu(pha_in,dh);
     pha_vfree=gather(pha_vfree);
     pha_vfree=bandlim_in.*pha_vfree+bandlim_ou.*pha;
     phi=exp(1i*pha_vfree);
   else
      phi=exp(1i*angle(es));
   end
end
C1=fftshift(fft2(fftshift(Amp.*phi)));
C1_o=C1.*H_AS;
H_s=fftshift(ifft2(fftshift(C1_o)));
hs=abs(H_s);
hs_am=sqrt(E.*(hs.^2)/sum(sum(hs.^2)));
H_s=hs_am.*exp(1i*angle(H_s));
Hl=Hl+H_s;
end
H=Hl((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4));
phase(:,:,j)=angle(phi);
[dph]=function_double_phase(H,M1,M2);
DPH(:,:,j)=dph;
imwrite(dph,['double_phase_',num2str(j),'hologram.bmp']);
end
%% filter
filter=zeros(nn,mm);
h=n*1.5;w=m*1.5;
filter(nn/2-h/2+1:nn/2+h/2,mm/2-w/2+1:mm/2+w/2)=1;
% figure,imshow(filter);
%% output
I_rec=zeros(nn/2,mm/2,3);
oz=150;
for j=1:3
% j=3;
hologram=exp(1i*2*pi*DPH(:,:,j));
hologram=padarray(hologram,[nn/4,mm/4]);
G=fftshift(fft2(fftshift(hologram)));
G1=filter.*G;
back=fftshift(ifft2(fftshift(G1)));
e=fftshift(fft2(fftshift(back)));
h_AS=exp(1i*k(j,1)*(-1)*oz.*sqrt(1-(lamda(j,1)*fx).^2-(lamda(j,1)*fy).^2)); 
e=ifftshift(ifft2(ifftshift(e.*h_AS)));
rec=abs(e).^2;
rec=rec((nn/4)+1:(nn*3/4),(mm/4)+1:(mm*3/4));
I_rec(:,:,j)=EE(j,1)*(rec/sum(sum(rec)));
end
%% 
figure,imshow(I_rec);
toc;