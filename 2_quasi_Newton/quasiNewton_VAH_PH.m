close all;clear;clc;
%% parameter
slm.pix = 3.74e-3;
slm.Nx = 384; slm.Ny = 384; 
opt.Nx = 2*slm.Nx; opt.Ny = 2*slm.Ny;
dh = 3.74e-3;
%% illumination pattern at SLM
slm.window = zeros(opt.Nx,opt.Ny);
slm.window( (opt.Nx/2-slm.Nx/2)+1 : (opt.Nx/2+slm.Nx/2) , (opt.Ny/2-slm.Ny/2)+1 : (opt.Ny/2+slm.Ny/2)) = ones(slm.Nx, slm.Ny);
opt.source = slm.window;
opt.in = ones(opt.Nx*2/3,opt.Ny*2/3);
opt.in = padarray(opt.in,[opt.Nx/6,opt.Ny/6]);
opt.ou = 1-opt.in;
%% Generate Dummy Data
load('object_grayscale');
obj = imresize(F1, [opt.Nx*2/3, opt.Ny*2/3]);
obj = padarray(obj,[opt.Nx/6,opt.Ny/6]);
Masks = obj;
E=sum(Masks(:));
El=0.5*E; Es=1.5*E;
F1=Masks((opt.Nx/2-opt.Nx/3)+1:(opt.Nx/2+opt.Nx/3),(opt.Ny/2-opt.Ny/3)+1:(opt.Ny/2+opt.Ny/3));
%% generation of the starting value
[LX,LY]=size(Masks);
pha=2*pi*rand(LX,LY);
pha=exp(1i*pha);
amp=opt.in.*sqrt(Masks)+opt.ou.*rand(LX,LY);
hologram=ifftshift(ifft2(ifftshift(amp .* pha)));
pha=angle(hologram);
%% optimization
matlab_options = optimoptions('fmincon','GradObj','on', ...
    'algorithm','interior-point','Hessian','lbfgs','FunValCheck','on','MaxFunEvals', 500 ,'MaxIter', 10,...
    'TolX', 1e-20, 'TolFun', 1e-15);
lb=-inf(opt.Nx*opt.Ny, 1);
ub=inf(opt.Nx*opt.Ny, 1);
times=30;
RMSE=zeros(times,1);NUM_PO=zeros(times,1);NUM_NE=zeros(times,1);
figure
for i=2:times
f=@(x)function_L2_PH(x, opt.source, opt.in, opt.Nx, opt.Ny, Masks);
[phase, loss] = fmincon(f,pha,[],[],[],[],lb,ub,[],matlab_options);
%% show result
objectField=opt.source.*exp(1i.*phase);
imagez=fftshift(fft2(fftshift(objectField)));  
amp=abs(imagez);
I=amp((opt.Nx/2-opt.Nx/3)+1:(opt.Nx/2+opt.Nx/3),(opt.Ny/2-opt.Ny/3)+1:(opt.Ny/2+opt.Ny/3)).^2;
I=E*I/sum(sum(I));
P=mod(angle(imagez),2*pi);
P=P((opt.Nx/2-opt.Nx/3)+1:(opt.Nx/2+opt.Nx/3),(opt.Ny/2-opt.Ny/3)+1:(opt.Ny/2+opt.Ny/3));
imshow(I);
%% vortex elimination
   Diff=double(I)-double(F1);
   MSE=gather(sum(Diff(:).^2)/numel(I));
   RMSE(i,1)=sqrt(MSE);
   diff_RMSE=RMSE(i,1)-RMSE(i-1,1);
   if abs(diff_RMSE)<0.007 && RMSE(i,1)>0.025
      pha_obj=angle(imagez);
      pha_in=opt.in.*pha_obj;
      [pha_vfree]=function_vortex_elimination_accegpu(pha_in,dh);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(pha_vfree,dh);
      pha_vfree=opt.in.*pha_vfree+opt.ou.*pha_obj;
      pha_vfree=exp(1i*pha_vfree);
      amp=sqrt(Es*(amp.^2)/sum(sum(amp.^2)));
      amp=opt.in.*sqrt(Masks)+opt.ou.*amp;
      hologram=ifftshift(ifft2(ifftshift(amp.* pha_vfree)));
      pha = angle(hologram);
   else
      pha=phase;
      phi_in=angle(imagez).*opt.in;
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(phi_in,dh);
   end
end
NUM=NUM_PO+NUM_NE;
rec.phase = reshape(pha, [opt.Nx, opt.Ny]);
hologram = exp(1i.*rec.phase);
%% reconstruction
Rec = fftshift(fft2(fftshift(opt.source.*hologram)));  
I = abs(Rec).^2;
I=I((opt.Nx/2-opt.Nx/3)+1:(opt.Nx/2+opt.Nx/3),(opt.Ny/2-opt.Ny/3)+1:(opt.Ny/2+opt.Ny/3));
I= E*I/sum(sum(I));
Phase = mod(angle(Rec),2*pi);
figure,imshow(I);