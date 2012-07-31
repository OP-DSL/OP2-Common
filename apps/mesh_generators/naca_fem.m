
close all;
clear all;

%
% use NACA0012 definition to construct aero upper surface
%

x = sin(linspace(0,pi/2,100)).^2;
y = 0.594689181*[0.298222773*sqrt(x) - 0.127125232*x    - 0.357907906*x.^2 ...
                                     + 0.291984971*x.^3 - 0.105174696*x.^4];
y(end)=0;

%figure
%plot(x,y,x,-y);

%
% apply complex square root transform
%

x0 = 0.0077;
z  = complex(x,y);
z  = sqrt(z-x0);
z  = [ -conj(fliplr(z)) z(2:end) ];

%figure
%plot(real(z),imag(z),'*-');

x = real(z); y = imag(z);

%
% generate grid in complex square root domain
%

%I = 20;
%J = 30;

%I = 100;
%J = 150;

%I = 200;
%J = 300;

%720K cells mesh
%I = 400;
%J = 600;

I = 1800;
J = 2200;

%I = 1200;
%J = 1800;

xb = linspace(-3,3,3*I+1);
yb = zeros(size(xb));

yb(I+2:2*I) = spline(x,y,xb(I+2:2*I));

xg = repmat(xb',1,J+1);
for i=1:3*I+1
  yg(i,:) = linspace(yb(i),3,J+1);
end

%figure
%plot(xg,yg,'k-',xg',yg','k-'); axis equal

%
% map it back to physical domain
%

z = complex(xg,yg).^2;
x = real(z); y = imag(z);

figure
plot(x,y,'k-',x',y','k-'); axis equal

%
% construct output file
%

nnodes   = (3*I+1)*(J+1) - (I+1);  % number of unique nodes
ncells   = 3*I*J;                  % number of cells
nffnodes = 3*I + 2*J;              % number of far-field nodes

node = zeros(3*I+1,J+1);
xn   = zeros(1,nnodes);
yn   = zeros(1,nnodes);

ff_node = zeros(1,nffnodes);
nff = 0;

for j = 1:J+1
  for i = 1:3*I+1
    k = i + (j-1)*(3*I+1) - (I+1);

    if (k>0)
      node(i,j) = k;
      xn(k) = x(i,j);
      yn(k) = y(i,j);

      if (i==1 || i==3*I+1 || j==J+1)
        nff = nff+1;
        ff_node(nff) = k;
      end
    else
      node(i,j) = 2*I+1 - i;
    end

  end
end

if (nff ~= nffnodes)
  disp('wrong number of far-field nodes');
end

cell_node = zeros(4,ncells);

wid = 1;
ic  = 0;

for j2 = 1:wid:J
 for i = 1:3*I
  for j = j2:min(j2+wid-1,J)
    ic = ic+1;
    cell_node(1,ic) = node(i  ,j);
    cell_node(2,ic) = node(i+1,j);
    cell_node(3,ic) = node(i  ,j+1);
    cell_node(4,ic) = node(i+1,j+1);
  end
 end
end

%
% switch to C numbering
%

cell_node = cell_node  - 1;
ff_node   = ff_node    - 1;

%
% output to grid.dat file
%

fid = fopen('FE_grid.dat','wt');

fprintf(fid,' %d %d %d \n',nnodes, ncells, nffnodes);

for n=1:nnodes
  fprintf(fid,' %f %f \n',xn(n),yn(n));
end

for n=1:ncells
  fprintf(fid,' %d %d %d %d \n',cell_node(1,n),cell_node(2,n),...
                                cell_node(3,n),cell_node(4,n) );
end

for n=1:nffnodes
  fprintf(fid,' %d \n',ff_node(n) );
end

fclose(fid);
