function [dph]=function_double_phase(H,M1,M2)
An=angle(H);
An=mod(An,2*pi);
Am=abs(H);
Am=Am/max(max(Am));
sita1=An-acos(Am);
sita2=An+acos(Am);
dph=M1.*sita1+M2.*sita2;
dph=mod(dph,2*pi);
dph=mat2gray(dph);
end