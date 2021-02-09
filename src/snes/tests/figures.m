clear all
close all

%load symnum data
summary_symnum
n7=N;
p7=reshape([ptap7(1:7);ptap7(22);ptap7(8:14);ptap7(23);ptap7(15:21);ptap7(24)],8,3)';
s7=reshape([ squ7(1:7); squ7(22); squ7(8:14); squ7(23); squ7(15:21); squ7(24)],8,3)';
o7=reshape([ opt7(1:7); opt7(22); opt7(8:14); opt7(23); opt7(15:21); opt7(24)],8,3)';

figure; bar(p7), xticklabels({'1','8','64'}), xlabel('Nodes'), ylabel('time (s)'), legend({'6C','12C','18C','24C','30C','36C','42C','6G'},'location', 'northeastoutside');
ax1=gca;
ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
set(ax2, 'XAxisLocation', 'top');
set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', {});
set(ax2, 'XTickLabel', n7);
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 1000 750])
saveas(gcf,'matmat_PtAP_7.png')
close(gcf)

figure; bar(s7), xticklabels({'1','8','64'}), xlabel('Nodes'), ylabel('time (s)'), legend({'6C','12C','18C','24C','30C','36C','42C','6G'},'location', 'northeastoutside');
ax1=gca;
ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
set(ax2, 'XAxisLocation', 'top');
set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', {});
set(ax2, 'XTickLabel', n7);
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 800 600])
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 1000 750])
saveas(gcf,'matmat_AtA_7.png')
close(gcf)

figure; bar(o7), xticklabels({'1','8','64'}), xlabel('Nodes'), ylabel('time (s)'), legend({'6C','12C','18C','24C','30C','36C','42C','6G'},'location', 'northeastoutside');
ax1=gca;
ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
set(ax2, 'XAxisLocation', 'top');
set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', {});
set(ax2, 'XTickLabel', n7);
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 800 600])
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 1000 750])
saveas(gcf,'matmat_AP_7.png')
close(gcf)

%load numonly data
summary_numonly
n7=N;
p7=reshape([ptap7(1:7);ptap7(22);ptap7(8:14);ptap7(23);ptap7(15:21);ptap7(24)],8,3)';
o7=reshape([ opt7(1:7); opt7(22); opt7(8:14); opt7(23); opt7(15:21); opt7(24)],8,3)';

figure; bar(p7), xticklabels({'1','8','64'}), xlabel('Nodes'), ylabel('time (s)'), legend({'6C','12C','18C','24C','30C','36C','42C','6G'},'location', 'northeastoutside');
ax1=gca;
ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
set(ax2, 'XAxisLocation', 'top');
set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', {});
set(ax2, 'XTickLabel', n7);
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 1000 750])
saveas(gcf,'matmat_PtAP_7_numonly.png')
close(gcf)

figure; bar(o7), xticklabels({'1','8','64'}), xlabel('Nodes'), ylabel('time (s)'), legend({'6C','12C','18C','24C','30C','36C','42C','6G'},'location', 'northeastoutside');
ax1=gca;
ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
set(ax2, 'XAxisLocation', 'top');
set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', {});
set(ax2, 'XTickLabel', n7);
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 800 600])
set( findall(gcf, '-property', 'fontsize'), 'fontsize', 20)
set(gca,'linewidth',1.5)
set(gcf,'Position',[100 100 1000 750])
saveas(gcf,'matmat_AP_7_numonly.png')
close(gcf)
