x = (linspace(0,1,100))';
y = (linspace(0,1,100))';
t = (0:0.02:0.99)';
u = @(x,y,t) exp(-t).*sin(2*pi.*x).*sin(2*pi.*y); 
for i = 1:100
    for j = 1:100
        for k= 1:50
            usol(i, j, k) = u(x(i), y(j), t(k));
        end
    end
end
filename = '2D-heat-equation-T=1.mat';

save(filename,'t','x','y','usol')

