function [xp, yp, obj, vobj,verror,count] = optimizer(X, fact_x, fact_y, maxiter)
verror = zeros(maxiter,1);
error = 1;
count = 0;

[obj_old, grad] = ObjectiveAndGradient(X);
nn = length(X);
xp = X(1:nn/2);
yp = X(nn/2+1:end);
vobj = [obj_old];
while ( (error > 1e-8) && (count < maxiter))
    xp = xp - fact_x * (grad(1,:));
    yp = yp - fact_y*(grad(2,:));
    X(1:nn/2) = xp;
    X(nn/2+1:end) = yp;
    [obj_new, grad] = ObjectiveAndGradient(X);
    vobj = [vobj, obj_new];
    count = count + 1;
    verror(count) = norm(obj_new-obj_old);
    error = verror(count);
    obj_old = obj_new;


end
obj = obj_new;
end


