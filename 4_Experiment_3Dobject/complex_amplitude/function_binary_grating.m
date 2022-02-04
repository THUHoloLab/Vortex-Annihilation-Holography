function [M1,M2]=function_binary_grating(nn,mm)

N=nn/2;M=mm/2;
M1=zeros(N,M);
M2=ones(N,M);
sma=zeros(N,M);
for x=1:N
    for y=1:M
        if mod(x+y,2)==1
            sma(x,y)=1;
        end
    end
end
for i=1:N
    for j=1:M
        M1(i,j)=sma(ceil(i),ceil(j));
    end
end
M2=M2-M1;
end