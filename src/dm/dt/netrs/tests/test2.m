clc;
clear;

% discretize only in parent and copy state for the daughter branches 

%Fix the water height and adjust the momentum (water slam test) 

heightp = linspace(-10,10,20); 
heightd = heightp; 

%preallocate evaluation matrices 
Roe = zeros(length(heightd),length(heightp));
Lax = Roe; 
Taylor = Roe; 
Exact  = Roe;
WaveType = Roe;
NegHeight = Roe; 
Fluvial   = Roe; 
for i = 1:length(heightp)
    for j = 1:length(heightd)
        hp = heightp(i);
        hd = heightd(j);
        command = "./ex1 -pu " + num2str(hp) + " -d1u "+ num2str(hd) + " -d2u " + num2str(hd)...
                + " -d1h 5 -ph 5 -d2h 5";
        jsystem(command,'noshell');
        %parent data 
        p = readtable("./ex1output/linearized_p.csv"); 
        Roe(i,j) = p.Roe; 
        Lax(i,j) = p.Lax; 
        Taylor(i,j) = p.Taylor; 
        Exact(i,j) = p.ExactStar; 
        NegHeight(i,j) = p.NegHeight;
        WaveType(i,j) = p.WaveType;
        Fluvial(i,j) = p.Fluvial; 
    end
end

%Build Plots 
figure; 
contourf(heightp,heightd,Roe);
colorbar

xlabel("Parent Momentum"); 
ylabel("Daughter Momentum"); 
title("Roe Error Estimate");

figure; 
contourf(heightp,heightd,Lax);
colorbar
xlabel("Parent Momentum"); 
ylabel("Daughter Momentum"); 
title("Lax Error Estimates"); 

figure; 
contourf(heightp,heightd,Taylor);
colorbar
xlabel("Parent Momentum"); 
ylabel("Daughter Momentum"); 
title("Taylor Error Estimates"); 

figure; 
contourf(heightp,heightd,Exact);
colorbar
xlabel("Parent Momentum"); 
ylabel("Daughter Momentum"); 
title("Star State Error For Linearized Solver");