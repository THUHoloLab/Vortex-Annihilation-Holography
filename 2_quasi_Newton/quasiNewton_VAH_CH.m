close all;clear;clc;
%% parameter
slm.pix = 3.74e-3;
slm.Nx = 256; slm.Ny = 256;
opt.Nx = 2*slm.Nx; opt.Ny = 2*slm.Ny;
dh = 3.74e-3;
%% illumination pattern at SLM
opt.source = 1;
slm.window = zeros(opt.Nx,opt.Ny);
slm.window( (opt.Nx/2-slm.Nx/2)+1 : (opt.Nx/2+slm.Nx/2) , (opt.Ny/2-slm.Ny/2)+1 : (opt.Ny/2+slm.Ny/2)) = ones(slm.Nx, slm.Ny);
opt.source = opt.source.*slm.window;
%% Generate Dummy Data
load('object_grayscale');
obj = imresize(F1, [opt.Nx, opt.Ny]);
Masks=obj; E=sum(Masks(:));
%% generation of the starting value
[LX,LY]=size(Masks);
pha=2*pi*rand(LX,LY);
%% optimization
matlab_options = optimoptions('fmincon','GradObj','on', ...
    'algorithm','interior-point','Hessian','lbfgs','FunValCheck','on','MaxFunEvals', 500 ,'MaxIter', 10,...
    'TolX', 1e-20, 'TolFun', 1e-15);
lb = -inf(opt.Nx*opt.Ny, 1);
ub = inf(opt.Nx*opt.Ny, 1);
f = @(x)function_L2_CH(x, opt.source, opt.Nx, opt.Ny, Masks);
times=30;
RMSE=zeros(times,1);NUM_PO=zeros(times,1);NUM_NE=zeros(times,1);
figure
for i=2:times
[phase, loss] = fmincon(f,pha,[],[],[],[],lb,ub,[],matlab_options);
%% show result
objectField = sqrt(Masks).*exp(1i.*phase);
imagez = opt.source.*fftshift(fft2(fftshift(objectField)));  
imagez = ifftshift(ifft2(ifftshift(imagez)));  
I = abs(imagez).^2;
I = E*I/sum(sum(I));
P = mod(angle(imagez),2*pi);
imshow(I);
%% vortex elimination
RMSE(i,1)=sqrt(loss/numel(Masks));
diff_RMSE=RMSE(i,1)-RMSE(i-1,1);
   if abs(diff_RMSE)<0.005 && RMSE(i,1)>0.025
      [pha_vfree]=function_vortex_elimination_accegpu(phase,dh);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(pha_vfree,dh);
      pha=pha_vfree;
   else
      pha=phase;
      phi_in=angle(imagez);
      [NUM_PO(i,1),NUM_NE(i,1)]=function_vortex_detection_accegpu(phi_in,dh);
   end
end
NUM=NUM_PO+NUM_NE;
rec.phase = reshape(pha, [opt.Nx, opt.Ny]);
objectField = sqrt(Masks).*exp(1i.*rec.phase);
hologram = opt.source.*fftshift(fft2(fftshift(objectField)));  
%% reconstruction
Rec = ifftshift(ifft2(ifftshift(hologeam)));  
I=abs(Rec).^2;
I=E*I/sum(sum(I));
Phase=mod(angle(Rec),2*pi);
figure,imshow(mat2gray(I));