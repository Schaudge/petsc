clear all;
close all;
clc ;

datadir = '~/OLCF_RUN_DATA/';
origdir = cd(datadir);
folders = dir;
[~, reindex] = sort(str2double(regexp({folders.name},'\d+','match','once')));
folders = folders(reindex);
numfolder = length(folders);

for k=1:numfolder-4
    curdirname = folders(k).name;
    cd(curdirname);
    foAF = dir('filtoutADDVAL*');
    foIF = dir('filtoutINSERT*');
    npF  = dir('nprocess*');
    psAF = dir('packsizeADDVAL*');
    psIF = dir('packsizeINSERT*');
    cprF = dir('cellprank*');
    vnF  = dir('vertnum*');
    cnF  = dir('cellnum*');
    oF   = dir('overlap*');
    vdTF = dir('vecdottime*');
    vdFF = dir('vecdotflops*');
    feoF = dir('feorder*');
    cpF  = dir('compnum*');
    gvsF = dir('globvecsize*');
    
    filtoutADDDat(k) = fopen(fullfile(foAF.name),'r');
    formatspec = '%f';
    commADDTime(k,:) = cell2mat(textscan(filtoutADDDat(k), formatspec));
    fclose(filtoutADDDat(k));
    
    filtoutINSERTDat(k) = fopen(fullfile(foIF.name),'r');
    %formatspec = '%10s %*s';
    formatspec = '%f';
    commINSERTTime(k,:) = cell2mat(textscan(filtoutINSERTDat(k), formatspec));
    fclose(filtoutINSERTDat(k));
    
    nProcessDat(k) = fopen(fullfile(npF.name),'r');
    formatspec = '%f';
    NRanks(k,:) = cell2mat(textscan(nProcessDat(k), formatspec));
    fclose(nProcessDat(k));
    
    packSizeADDDat(k) = fopen(fullfile(psAF.name),'r');
    formatspec = '%f';
    MPIPackSizeADD(k,:) = cell2mat(textscan(packSizeADDDat(k), formatspec));
    fclose(packSizeADDDat(k));
    
    packSizeINSERTDat(k) = fopen(fullfile(psIF.name),'r');
    formatspec = '%f';
    MPIPackSizeINSERT(k,:) = cell2mat(textscan(packSizeINSERTDat(k), formatspec));
    fclose(packSizeINSERTDat(k));
    
    cellPrankDat(k) = fopen(fullfile(cprF.name),'r');
    formatspec = '%f';
    CellPRank(k,:) = cell2mat(textscan(cellPrankDat(k), formatspec));
    fclose(cellPrankDat(k));
    
    VertNumDat(k) = fopen(fullfile(vnF.name),'r');
    formatspec = '%f';
    VertNum(k,:) = cell2mat(textscan(VertNumDat(k), formatspec));
    fclose(VertNumDat(k));
    
    CellNumDat(k) = fopen(fullfile(cnF.name),'r');
    formatspec = '%f';
    CellNum(k,:) = cell2mat(textscan(CellNumDat(k), formatspec));
    fclose(CellNumDat(k));
    
    OverlapDat(k) = fopen(fullfile(oF.name),'r');
    formatspec = '%f';
    Overlap(k,:) = cell2mat(textscan(OverlapDat(k), formatspec));
    fclose(OverlapDat(k));
    
    VecDotTDat(k) = fopen(fullfile(vdTF.name),'r');
    formatspec = '%f';
    VecDotT(k,:) = cell2mat(textscan(VecDotTDat(k), formatspec));
    fclose(VecDotTDat(k));
    
    VecDotFlopsDat(k) = fopen(fullfile(vdFF.name),'r');
    formatspec = '%f';
    VecDotFlops(k,:) = cell2mat(textscan(VecDotFlopsDat(k), formatspec));
    fclose(VecDotFlopsDat(k));
    
    FEOrderDat(k) = fopen(fullfile(feoF.name),'r');
    formatspec = '%f';
    FEOrder(k,:) = cell2mat(textscan(FEOrderDat(k), formatspec));
    fclose(FEOrderDat(k));
    
    NumCompDat(k) = fopen(fullfile(cpF.name),'r');
    formatspec = '%f';
    NumComp(k,:) = cell2mat(textscan(NumCompDat(k), formatspec));
    fclose(NumCompDat(k));
    
    GVSDat(k) = fopen(fullfile(gvsF.name),'r');
    formatspec = '%f';
    GVS(k,:) = cell2mat(textscan(GVSDat(k), formatspec));
    fclose(GVSDat(k));
    
    cd(folders(k).folder);
end
clear filtoutADDDAt filtoutINSERTDat nProcessDat packSizeADDDat packSizeINSERTDat
clear cellPrankDat VertNumDat CellNumDat OverlapDat VecDotTDat VecDotFlopsDat
clear FEOrderDat NumCompDat GVSDat foAF foIF npF psAF psIF cprF vnF cnF oF vdTF
clear vdFF feoF cpF gvsF formatspec reindex numfoler datadir rawADDTimings rawINSERTTimings

[numarr, arrlengths] = size(commADDTime);
commADDTime = commADDTime/100;
commINSERTTime = commINSERTTime/100;
VecDotT = VecDotT/100;
MPIPackSizeADD = MPIPackSizeADD/100;
MPIPackSizeINSERT = MPIPackSizeINSERT/100;
Psize = VertNum(1)*NumComp(1);



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
for entry=6:numarr+2
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
plot(NRanks(:,1),(commADDTime(:,entry-5).*NRanks(:,1))./MPIPackSizeADD(:,entry-5), 'x-',...
    'MarkerSize', 10, 'DisplayName',...
    ['ADD RANKS = ', num2str(6*(2^(entry-1-5)))]);
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