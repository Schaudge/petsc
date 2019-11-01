clear all;
close all;
clc ;

datadir = '~/OLCF_RUN_DATA/';
origdir = cd(datadir);
folders = dir;
numfolder = length(folders);
i = 1;

for k=1:numfolder
    if (folders(k).name(1) == '.')
        continue
    end
    if (~(folders(k).name(1:5) == 'DEBUG'))
        continue
    end
    curdirname = folders(k).name;
    cd(curdirname);
    foAF = dir('filtoutADDVAL*');
    foIF = dir('filtoutINSERT*');
    npF  = dir('nprocess*');
    psAF = dir('packsizeADDVAL*');
    psIF = dir('packsizeINSERT*');
    cnF  = dir('cellnum*');
    nnF  = dir('nodenum*');
    oF   = dir('overlap*');
    vdTF = dir('vecdottime*');
    vdFF = dir('vecdotflops*');
    feoF = dir('feorder*');
    cpF  = dir('compnum*');
    gvsF = dir('globvecsize*');
    sfpIF = dir('sfpackINSERT*');
    sfupIF = dir('sfunpackINSERT*');
    sfpAF = dir('sfpackADDVAL*');
    sfupAF = dir('sfunpackADDVAL*');
    sfmAF  = dir('sfmessADDVAL*');
    sfmIF  = dir('sfmessINSERT*');
    
    commADDTime(i,:) = transpose(load(foAF.name));
    commINSERTTime(i,:) = transpose(load(foIF.name));
    NRanks(i,:) = transpose(load(npF.name));
    MPIPackSizeADD(i,:) = transpose(load(psAF.name));
    MPIPackSizeINSERT(i,:) =transpose(load(psIF.name));
    NodeNum(i,:) = transpose(load(psIF.name));
    CellNum(i,:) = transpose(load(cnF.name));
    Overlap(i,:) = transpose(load(oF.name));
    VecDotT(i,:) = transpose(load(vdTF.name));
    VecDotFlops(i,:) = transpose(load(vdFF.name));
    FEOrder(i,:) = transpose(load(feoF.name));
    NumComp(i,:) =transpose(load(cpF.name)); 
    GVS(i,:) = transpose(load(gvsF.name));
    SFPINSERT(i,:) = transpose(load(sfpIF.name));
    SFUPINSERT(i,:) = transpose(load(sfupIF.name));
    SFPADDVAL(i,:) = transpose(load(sfpAF.name));
    SFUPADDVAL(i,:) = transpose(load(sfupAF.name));
    SFMESSADDVAL(i,:) = transpose(load(sfmAF.name));
    SFMESSINSERT(i,:) = transpose(load(sfmIF.name));
    
    i = i + 1;
    cd(folders(i).folder);
end
clear foAF foIF npF psAF psIF cprF vnF cnF oF vdTF sfmAF sfmIF
clear vdFF feoF cpF gvsF formatspec reindex numfoler datadir rawADDTimings rawINSERTTimings
clear SFPINSERTDat SFUPINSERTDat SFPADDVALDat SFUPADDVALDat sfpIF sfupIf sfpAF sfupAF

[~, idx] = sort(NRanks(:,1));

commADDTime = commADDTime(idx,:);
commINSERTTime = commINSERTTime(idx,:);
NRanks = NRanks(idx,:);
MPIPackSizeADD = MPIPackSizeADD(idx,:);
MPIPackSizeINSERT = MPIPackSizeINSERT(idx,:);
NodeNum = NodeNum(idx,:);
CellNum = CellNum(idx,:);
Overlap = Overlap(idx,:);
VecDotT = VecDotT(idx,:);
VecDotFlops = VecDotFlops(idx,:);
FEOrder = FEOrder(idx,:);
NumComp = NumComp(idx,:);
GVS = GVS(idx,:);
SFPINSERT = SFPINSERT(idx,:);
SFUPINSERT = SFUPINSERT(idx,:);
SFPADDVAL = SFPADDVAL(idx,:);
SFUPADDVAL = SFUPADDVAL(idx,:);
SFMESSADDVAL = SFMESSADDVAL(idx,:);
SFMESSINSERT = SFMESSINSERT(idx,:);

[numarr, arrlengths] = size(commADDTime);
%commADDTime = commADDTime/100;
%commINSERTTime = commINSERTTime/100;
%VecDotT = VecDotT/100;
%MPIPackSizeADD = MPIPackSizeADD/100;
%MPIPackSizeINSERT = MPIPackSizeINSERT/100;
Psize = NodeNum(1)*NumComp(1);

% n = 1; % average every n values, then average vectors
% commADDTime = arrayfun(@(i) mean(commADDTime(i:i+n-1)),1:n:length(commADDTime)-n+1)';
% commINSERTTime = arrayfun(@(i) mean(commINSERTTime(i:i+n-1)),1:n:length(commINSERTTime)-n+1)';
% CellPRank = arrayfun(@(i) mean(CellPRank(i:i+n-1)),1:n:length(CellPRank)-n+1)';
% NRanks = arrayfun(@(i) mean(NRanks(i:i+n-1)),1:n:length(NRanks)-n+1)';
% MPIPackSizeADD = arrayfun(@(i) mean(MPIPackSizeADD(i:i+n-1)),1:n:length(MPIPackSizeADD)-n+1)';
% MPIPackSizeINSERT = arrayfun(@(i) mean(MPIPackSizeINSERT(i:i+n-1)),1:n:length(MPIPackSizeINSERT)-n+1)';
% VecDotT = arrayfun(@(i) mean(VecDotT(i:i+n-1)),1:n:length(VecDotT)-n+1)';
% VecDotFlops = arrayfun(@(i) mean(VecDotFlops(i:i+n-1)),1:n:length(VecDotFlops)-n+1)';

figure(1)
cmap = hsv(15);
set(gcf,'Position',[0,0,1400,700])
posvecupper = [0.05 0.55 0.92 0.4];
posveclower = [0.05 0.05 0.92 0.4];
axes('Units', 'normalized', 'Position', posvecupper);
grid on;
title('Timings NC = 1','fontweight','bold');
xlabel('PetscSpace Degree','fontweight','bold');
ylabel('Time [s]','fontweight','bold');
axtop = gca;
axes('Units', 'normalized', 'Position', posveclower);
grid on;
title('Timings NC = 3','fontweight','bold');
xlabel('PetscSpaceDegree','fontweight','bold');
ylabel('Time [s]','fontweight','bold');
axbot = gca;
%set(axtop,'xscale','log');
set(axtop,'linewidth',1.2);
%set(axbot,'yscale','log');
set(axbot,'linewidth',1.2);
for entry=1:numarr
    axes(axtop);
    hold on;
%     plot(FEOrder(entry,1:5),commADDTime(entry,1:5)./(double(GVS(entry,1:5))), 'x-',...
%         'MarkerSize', 10, 'DisplayName',...
%         ['ADD RANKS = ', num2str(6*(2^(entry-1)))],...
%         'Color', cmap(entry,:));
plot(NRanks(:,1),(commADDTime(:,entry).*NRanks(:,1))./MPIPackSizeADD(:,entry), 'x-',...
    'MarkerSize', 10, 'DisplayName',...
    ['ADD RANKS = ', num2str(6*(2^(entry-1)))]);
%     plot(FEOrder(entry,1:5),commINSERTTime(entry,1:5), '.-',...
%         'MarkerSize', 10, 'DisplayName',...
%         ['INS RANKS = ', num2str(6*(2^(entry-1)))],...
%         'Color', cmap(entry,:));
    hold off;
    subplot(axtop);
    
    axes(axbot);
    hold on;
%     plot(FEOrder(entry,6:end),commADDTime(entry,6:end), 'x-',...
%         'MarkerSize', 10, 'DisplayName',...
%         ['ADD RANKS = ', num2str(6*(2^(entry-1)))],...
%         'Color', cmap(entry,:));
plot(NRanks(:,1),(commADDTime(:,entry).*NRanks(:,1))./MPIPackSizeADD(:,entry), 'x-',...
    'MarkerSize', 10, 'DisplayName',...
    ['ADD RANKS = ', num2str(6*(2^(entry-1+5)))]);
%     plot(FEOrder(entry,1:5),commINSERTTime(entry,1:5)./(double(GVS(entry,1:5))), '.-',...
%         'MarkerSize', 10, 'DisplayName',...
%         ['INS RANKS = ' , num2str(6*(2^(entry-1)))],...
%         'Color', cmap(entry,:));
    hold off;
    subplot(axbot);
end
legend(axtop, 'Location', 'eastoutside');
legend(axbot, 'Location', 'eastoutside');
if (0)
    figure(2)
    set(gcf,'Position',[100,200,1200,700])
    subplot(totalm,totaln,1)
    hold on
    grid on
    plot(commADDTime)
    hold off
    title('comm add')
    legend({'ADD VALUES'}, 'location','southeast')
    dim = [0.6 0.05 0.5 0.5];
    str = {['Cells: ', num2str(CellNum(1))],...
        ['Fields: ', num2str(NumComp(1))],...
        ['PetscSpace Degree: ', num2str(FEOrder(1))],...
        ['Tot Prob size: ', num2str(Psize)]};
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
    %set(gca,'xscale','log')
    %set(gca,'yscale','log')
    xlabel('Ranks')
    ylabel('Time [s]')
    subplot(totalm,totaln,2)
    hold on
    grid on
    plot(commINSERTTime)
    hold off
    title('comm insert')
    legend({'ADD VALUES'}, 'location','southeast')
    %set(gca,'xscale','log')
    %set(gca,'yscale','log')
    xlabel('Ranks')
    ylabel('FLOPS')
end
cd(origdir);