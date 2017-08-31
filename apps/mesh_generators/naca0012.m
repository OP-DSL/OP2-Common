%
% use NACA0012 definition to construct airfoil upper surface
%

function naca0012(grid)

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

I = 800;
J = 1200;

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

%figure
%plot(x,y,'k-',x',y','k-'); axis equal

if (nargin > 0 && strcmpi(grid, 'old'))

%
% construct old-style output file
%

nnodes = (3*I+1)*(J+1);
ncells =  3*I*J;
nedges = (3*I+1)*J + 3*I*J + 2*I; % right/left, up/down, bottom line

node = zeros(3*I+1,J+1);
xn   = zeros(1,nnodes);
yn   = zeros(1,nnodes);

for j = 1:J+1
  for i = 1:3*I+1
    k = i + (j-1)*(3*I+1);
    node(i,j) = k;
    xn(k) = x(i,j);
    yn(k) = y(i,j);
  end
end

cell      = zeros(3*I,J);
cell_node = zeros(4,ncells);

for j = 1:J
  for i = 1:3*I
    k = i + (j-1)*(3*I);
    cell(i,j) = k;
    cell_node(1,k) = node(i  ,j);
    cell_node(2,k) = node(i+1,j);
    cell_node(3,k) = node(i+1,j+1);
    cell_node(4,k) = node(i  ,j+1);
  end
end

edge_node = zeros(2,nedges);
edge_cell = zeros(2,nedges);
boun      = zeros(1,nedges);

ie = 0;

for j = 1:J
  for i = 1:3*I

%
% left
%
    ie = ie+1;
    edge_node(1,ie) = node(i,j);
    edge_node(2,ie) = node(i,j+1);
    edge_cell(1,ie) = cell(i,j);
    if (i==1)
      edge_cell(2,ie) = 0;                 %far-field
      boun(ie)        = 2;
    else
      edge_cell(2,ie) = cell(i-1,j);
      boun(ie)        = 0;
    end

%
% down for bottom row
%

    if (j==1 && i<=2*I)
      ie = ie+1;
      edge_node(1,ie) = node(i,j);
      edge_node(2,ie) = node(i+1,j);
      if (i<=I)
        edge_cell(1,ie) = cell(3*I+1-i,j); % cut-line
        boun(ie)        = 0;
      else
        edge_cell(1,ie) = cell(i,j);       % airfoil
        boun(ie)        = 1;
      end
      edge_cell(2,ie) = cell(i,j);
    end

%
% up
%
    ie = ie+1;
    edge_node(1,ie) = node(i,j+1);
    edge_node(2,ie) = node(i+1,j+1);
    edge_cell(1,ie) = cell(i,j);
    if (j==J)
      edge_cell(2,ie) = 0;
      boun(ie)        = 2;
    else
      edge_cell(2,ie) = cell(i,j+1);
      boun(ie)        = 0;
    end

%
% last to right
%
    if(i==3*I)
      ie = ie+1;
      edge_node(1,ie) = node(i+1,j+1);
      edge_node(2,ie) = node(i+1,j);
      edge_cell(1,ie) = cell(i,j);
      edge_cell(2,ie) = 0;                 % far-field
      boun(ie)        = 2;
    end

  end
end

%
% output to grid.dat file
%

fid = fopen('grid.dat','wt');

fprintf(fid,' %d %d %d\n',nnodes, ncells, nedges);

for n=1:nnodes
  fprintf(fid,' %f %f \n',xn(n),yn(n));
end

for n=1:ncells
  fprintf(fid,' %d %d %d %d \n',cell_node(1,n),cell_node(2,n),...
                                cell_node(3,n),cell_node(4,n));
end

for n=1:nedges
  fprintf(fid,' %d %d %d %d %d \n',edge_node(1,n),edge_node(2,n),...
                                   edge_cell(1,n),edge_cell(2,n),...
                                   boun(n));
end

fclose(fid);

else % (nargin > 0 && strcmpi(grid, 'old'))

%
% construct new-style output file
%

nnodes  = (3*I+1)*(J+1);
ncells  =  3*I*J;
nedges  = (3*I-1)*J + 3*I*(J-1) + I; % interior right/left, up/down, cut-line
nbedges = I + (3*I) + 2*J;           % airfoil and far-field boundaries

node = zeros(3*I+1,J+1);
xn   = zeros(1,nnodes);
yn   = zeros(1,nnodes);

for j = 1:J+1
  for i = 1:3*I+1
    k = i + (j-1)*(3*I+1);
    node(i,j) = k;
    xn(k) = x(i,j);
    yn(k) = y(i,j);
  end
end

cell      = zeros(3*I,J);
cell_node = zeros(4,ncells);

for j = 1:J
  for i = 1:3*I
    k = i + (j-1)*(3*I);
    cell(i,j) = k;
    cell_node(1,k) = node(i  ,j);
    cell_node(2,k) = node(i+1,j);
    cell_node(3,k) = node(i+1,j+1);
    cell_node(4,k) = node(i  ,j+1);
  end
end

edge_node = zeros(2,nedges);
edge_cell = zeros(2,nedges);

ie = 0;

for j = 1:J
  for i = 1:3*I
    if (j==1 && i<=I)
      ie = ie+1;
      edge_node(1,ie) = node(i  ,j);
      edge_node(2,ie) = node(i+1,j);
      edge_cell(1,ie) = cell(3*I+1-i,j);
      edge_cell(2,ie) = cell(i  ,j);
    end

    if (j<J)
      ie = ie+1;
      edge_node(1,ie) = node(i  ,j+1);
      edge_node(2,ie) = node(i+1,j+1);
      edge_cell(1,ie) = cell(i  ,j  );
      edge_cell(2,ie) = cell(i  ,j+1);
    end

    if (i<3*I)
      ie = ie+1;
      edge_node(1,ie) = node(i+1,j  );
      edge_node(2,ie) = node(i+1,j+1);
      edge_cell(1,ie) = cell(i+1,j  );
      edge_cell(2,ie) = cell(i  ,j  );
    end
  end
end

if (ie ~= nedges)
  disp('wrong number of new edges');
end

bedge_node = zeros(2,nbedges);
bedge_cell = zeros(nbedges);
bound      = zeros(nbedges);

ie = 0;

for i = I+1:2*I
  ie = ie+1;
  bedge_node(1,ie) = node(i+1,1);
  bedge_node(2,ie) = node(i  ,1);
  bedge_cell(ie)   = cell(i  ,1);
  bound(ie)        = 1;
end

for j = 1:J
  ie = ie+1;
  bedge_node(1,ie) = node(3*I+1,j+1);
  bedge_node(2,ie) = node(3*I+1,j  );
  bedge_cell(ie)   = cell(3*I  ,j  );
  bound(ie)        = 2;
end

for i = 3*I:-1:1
  ie = ie+1;
  bedge_node(1,ie) = node(i  ,J+1);
  bedge_node(2,ie) = node(i+1,J+1);
  bedge_cell(ie)   = cell(i  ,J  );
  bound(ie)        = 2;
end

for j = J:-1:1
  ie = ie+1;
  bedge_node(1,ie) = node(1,j  );
  bedge_node(2,ie) = node(1,j+1);
  bedge_cell(ie)   = cell(1,j  );
  bound(ie)        = 2;
end

if (ie ~= nbedges)
  disp('wrong number of new bedges');
end

%
% switch to C numbering
%

cell_node  = cell_node  - 1;
edge_node  = edge_node  - 1;
edge_cell  = edge_cell  - 1;
bedge_node = bedge_node - 1;
bedge_cell = bedge_cell - 1;

%
% output to grid.dat file
%

fid = fopen('new_grid.dat','wt');

fprintf(fid,' %d %d %d %d \n',nnodes, ncells, nedges, nbedges);

for n=1:nnodes
  fprintf(fid,' %f %f \n',xn(n),yn(n));
end

for n=1:ncells
  fprintf(fid,' %d %d %d %d \n',cell_node(1,n),cell_node(2,n),...
                                cell_node(3,n),cell_node(4,n) );
end

for n=1:nedges
  fprintf(fid,' %d %d %d %d \n',edge_node(1,n),edge_node(2,n),...
                                edge_cell(1,n),edge_cell(2,n) );
end

for n=1:nbedges
  fprintf(fid,' %d %d %d %d \n',bedge_node(1,n),bedge_node(2,n),...
                                bedge_cell(n),  bound(n) );
end

fclose(fid);

end % (nargin > 0 && strcmpi(grid, 'old'))

end % function naca0012(grid)
