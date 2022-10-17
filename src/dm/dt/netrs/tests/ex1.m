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
FluxExact = Roe;

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
        FluxExact(i,j) = p.ExactFlux;
    end
end

%Build Plots 
figure; 
contourf(heightp,heightd,Roe);
colorbar

ylabel("Parent Height"); 
xlabel("Daughter Height"); 
title("Roe Error Estimate");
saveas(gcf,'Roe_Ex1.jpg');

figure; 
contourf(heightp,heightd,Lax);
colorbar
ylabel("Parent Height"); 
xlabel("Daughter Height"); 
title("Lax Error Estimates"); 
saveas(gcf,'Lax_Ex1.jpg');

figure; 
contourf(heightp,heightd,Taylor);
colorbar
ylabel("Parent Height"); 
xlabel("Daughter Height"); 
title("Taylor Error Estimates"); 
saveas(gcf,'Taylor_Ex1.jpg');

figure; 
contourf(heightp,heightd,Exact);
colorbar
ylabel("Parent Height"); 
xlabel("Daughter Height"); 
title("Star State Error For Linearized Solver");
saveas(gcf,'ExactStar_Ex1.jpg');

figure; 
contourf(heightp,heightd,FluxExact);
colorbar
ylabel("Parent Height"); 
xlabel("Daughter Height"); 
title("Flux Error For Linearized Solver");
saveas(gcf,'ExactFlux.jpg'); 