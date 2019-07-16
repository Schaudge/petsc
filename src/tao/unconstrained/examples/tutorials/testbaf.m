global rc
xp = [-11.670068025781585  -2.070088940945681   7.681471596530399  -7.813649695667054 2.023612862668205  10.188667586619095  -7.970047969128254   0.878261077398502 9.688667586619095];

yp = [20.679513713636304  22.512288331717748  20.694862425798604  31.891770499954614 30.346545805956886  30.089536503152964  37.931334294198543  36.961256420659133 29.589536503152964];
nc =length(xp)
  r = rc
   figure(1)
    for ii=1:nc
        GG{ii} = cylinder(rc(ii),xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
        GG_r{ii} = cylinder_r(rc(ii),xp(ii),(yp(ii) - 2*ys));
        GG_r{ii} = curvquad(GG_r{ii},'ptr',N,10);
       quiver(real(GG{ii}.x),imag(GG{ii}.x),real(GG{ii}.nx),imag(GG{ii}.nx));
        hold on
        plot(real(GG{ii}.x),imag(GG{ii}.x),'k','LineWidth',2)
        hold on
       quiver(real(GG_r{ii}.x),imag(GG_r{ii}.x),real(GG_r{ii}.nx),imag(GG_r{ii}.nx));
        hold on
        plot(real(GG_r{ii}.x),imag(GG_r{ii}.x),'k','LineWidth',2)
        hold on
    end
   
    plot(x0c+rj*sin(thn),y0c+rj*cos(thn),'k','LineWidth',4)
    %in thick red the objective region
    plot(x0c+rj*sin(pi/6:0.1:pi/3),y0c+rj*cos(pi/6:0.1:pi/3),'r','LineWidth',3) %remets pi/3
    %source of incident wave
    plot(xs,ys,'ks','MarkerSize',8,'LineWidth',2)
    hold off
    axis equal