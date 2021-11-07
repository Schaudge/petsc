clc;
clear;

% discretize only in parent and copy state for the daughter branches 

%Fix daughters to be the same and fix momentum to be zero (Note that we are
%always fluvial in this case%

heightp = linspace(0.05,10,20); 
heightd = heightp; 

%preallocate evaluation matrices 
Roe = zeros(length(heightd),length(heightp));
Lax = Roe; 
Taylor = Roe; 
Exact  = Roe;
WaveType = Roe;
NegHeight = Roe; 

for i = 1:length(heightp)
    for j = 1:length(heightd)
        hp = heightp(i);
        hd = heightd(j);
        command = "./ex1 -ph " + num2str(hp) + " -d1h "+ num2str(hd) + " -d2h " + num2str(hd)...
                + " -d1u 0 -pu 0 -d2u 0";
        jsystem(command,'noshell');
        %parent data 
        p = readtable("./ex1output/linearized_p.csv"); 
        Roe(i,j) = p.Roe; 
        Lax(i,j) = p.Lax; 
        Taylor(i,j) = p.Taylor; 
        Exact(i,j) = p.ExactStar; 
        NegHeight(i,j) = p.NegHeight;
        WaveType(i,j) = p.WaveType;
    end
end

%Build Plots 
figure; 
contourf(heightp,heightd,Roe);
colorbar

xlabel("Parent Height"); 
ylabel("Daughter Height"); 
title("Roe Error Estimate");

figure; 
contourf(heightp,heightd,Lax);
colorbar
xlabel("Parent Height"); 
ylabel("Daughter Height"); 
title("Lax Error Estimates"); 

figure; 
contourf(heightp,heightd,Taylor);
colorbar
xlabel("Parent Height"); 
ylabel("Daughter Height"); 
title("Taylor Error Estimates"); 

figure; 
contourf(heightp,heightd,Exact);
colorbar
xlabel("Parent Height"); 
ylabel("Daughter Height"); 
title("Star State Error For Linearized Solver");