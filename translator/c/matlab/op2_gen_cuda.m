%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CUDA code generator
%
% This routine is called by op2 which parses the input files
%
% It produces a file xxx_kernel.cu for each kernel,
% plus a master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function op2_gen_cuda(master,date,consts,kernels)

global dims idxs typs indtyps inddims

OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

accsstring = {'OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN'};

any_soa = 0;
for nk = 1:length(kernels)
    any_soa = any_soa || sum(kernels{nk}.soaflags);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  create new kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for nk = 1:length(kernels)

  name  = kernels{nk}.name;

  nargs = kernels{nk}.nargs;
  dims  = kernels{nk}.dims;
  maps  = kernels{nk}.maps;
  vars  = kernels{nk}.vars;
  typs  = kernels{nk}.typs;
  accs  = kernels{nk}.accs;
  idxs  = kernels{nk}.idxs;
  inds  = kernels{nk}.inds;
  soaflags = kernels{nk}.soaflags;

  ninds   = kernels{nk}.ninds;
  inddims = kernels{nk}.inddims;
  indaccs = kernels{nk}.indaccs;
  indtyps = kernels{nk}.indtyps;
  invinds = kernels{nk}.invinds;

  if (~isempty(find(idxs<0 & maps == 3, 1)))
      unique_args = 1;
      vec_counter = 1;
      vectorised = [];
      new_dims = {};
      new_maps = [];
      new_vars = {};
      new_typs = {};
      new_accs = [];
      new_idxs = [];
      new_inds = [];
      new_soaflags = [];
      for m = 1:nargs
          if (idxs(m)<0 && maps(m) == 3)
              if m>1
                unique_args = [unique_args length(new_dims)+1];
              end
              temp = {};
              temp(1:-1*idxs(m)) = vars(m);
              new_vars = [new_vars temp];
              temp = {};
              temp(1:-1*idxs(m)) = typs(m);
              new_typs = [new_typs temp];
              temp = {};
              temp(1:-1*idxs(m)) = dims(m);
              new_dims = [new_dims temp];
              new_maps = [new_maps maps(m)*ones(1,-1*idxs(m))];
              new_soaflags = [new_soaflags zeros(1,-1*idxs(m))];
              new_accs = [new_accs accs(m)*ones(1,-1*idxs(m))];
              new_idxs = [new_idxs 0:(-1*idxs(m) -1)];
              new_inds = [new_inds inds(m)*ones(1,-1*idxs(m))];
              vectorised = [vectorised ones(1,-1*idxs(m)) * vec_counter];
              vec_counter = vec_counter + 1;
          else
              if m>1
                unique_args = [unique_args length(new_dims)+1];
              end
              new_dims = [new_dims dims(m)];
              new_maps = [new_maps maps(m)];
              new_accs = [new_accs accs(m)];
              new_soaflags = [new_soaflags soaflags(m)];
              new_idxs = [new_idxs idxs(m)];
              new_inds = [new_inds inds(m)];
              new_vars = [new_vars vars(m)];
              new_typs = [new_typs typs(m)];
              vectorised = [vectorised 0];
          end
      end
      dims = new_dims;
      maps = new_maps;
      accs = new_accs;
      idxs = new_idxs;
      inds = [new_inds];
      vars = new_vars;
      typs = new_typs;
      soaflags = new_soaflags;
      nargs = length(vectorised);
      for i = 1:ninds
          invinds(i) = find(inds == i,1);
      end
  else
      vectorised = zeros(1,nargs);
      unique_args = 1:nargs;
  end

  cumulative_indirect_index = -1*ones(1,nargs);
  cumulative_indirect_index(find(maps == 3)) = 0:(sum(maps==3)-1);

%
% set two logicals
%

  ind_inc = max(maps==OP_MAP & accs==OP_INC)  > 0;
  reduct  = max(maps==OP_GBL & accs~=OP_READ) > 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  start with CUDA kernel function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  file = strvcat('// user function         ',' ',...
                 '__device__               ',...
                ['#include "' name '.h"'],' ',' ',...
                 '// CUDA kernel function',' ',...
                ['__global__ void op_cuda_' name '(']);

  for m = 1:ninds
    line = '  INDTYP *ind_ARG,';
    file = strvcat(file,rep(line,m));
  end

  if (ninds>0)
    file = strvcat(file,'  int   *ind_map,');
    file = strvcat(file,'  short *arg_map,');
  end

  for m = 1:nargs
    if (maps(m)==OP_GBL && accs(m)==OP_READ)
      line = '  const TYP *ARG,';    % declared const for performance
      file = strvcat(file,rep(line,m));
    elseif (maps(m)==OP_ID && ninds>0)
      line = '  TYP *ARG,';
      file = strvcat(file,rep(line,m));
    elseif (maps(m)==OP_GBL || maps(m)==OP_ID)
      line = '  TYP *ARG,';
      file = strvcat(file,rep(line,m));
    end
  end

  if (ninds>0)
    file = strvcat(file,'  int   *ind_arg_sizes,',...
                        '  int   *ind_arg_offs,',...
                        '  int    block_offset,',...
                        '  int   *blkmap,      ',...
                        '  int   *offset,      ',...
                        '  int   *nelems,      ',...
                        '  int   *ncolors,     ',...
                        '  int   *colors,      ',...
                        '  int   nblocks,     ',...
                        '  int   set_size) {    ',' ');
  else
    file = strvcat(file,'  int   offset_s,   ',...
                        '  int   set_size ) {',' ');
  end

  for m = 1:nargs
    if (maps(m)==OP_GBL && accs(m)~=OP_READ)
      line = '  TYP ARG_l[DIM];';
      file = strvcat(file,rep(line,m));
      if (accs(m)==OP_INC)
        line = '  for (int d=0; d<DIM; d++) ARG_l[d]=ZERO_TYP;';
      else
        line = ...
        '  for (int d=0; d<DIM; d++) ARG_l[d]=ARG[d+blockIdx.x*DIM];';
      end
      file = strvcat(file,rep(line,m));
    elseif (maps(m)==OP_MAP && accs(m)==OP_INC)
      line = '  TYP ARG_l[DIM];';
      file = strvcat(file,rep(line,m));
    elseif (ninds==0 && maps(m)==OP_ID && ~strcmp(dims{m},'1') && ~soaflags(m))
      line = '  TYP ARG_l[DIM];';
      file = strvcat(file,rep(line,m));
    end
  end

  for  m = 1:ninds
    if (sum(inds==m)>1 & vectorised(find(inds==m)))
      if (indaccs(m) == OP_INC)
        line = sprintf('  INDTYP *ARG_vec[%d] = {',max(idxs(inds==m))+1);
        file = strvcat(file,rep(line,m));
        line = '';
        for n = 1:nargs
          if (inds(n) == m)
            file = strvcat(file,line);
            line = rep('    ARG_l,',n);
          end
        end
        file = strvcat(file,line(1:end-1));
        line = '  };';
        file = strvcat(file,line);
      else
        line = sprintf('  INDTYP *ARG_vec[%d];',max(idxs(inds==m))+1);
        file = strvcat(file,rep(line,m));
      end
    end
  end

%
% lengthy code for general case with indirection
%
  if (ninds>0)
    file = strvcat(file,' ');
    for m = 1:ninds
      line = '  __shared__ int   *ind_ARG_map, ind_ARG_size;';
      file = strvcat(file,rep(line,m));
    end
    for m = 1:ninds
      line = '  __shared__ INDTYP *ind_ARG_s;';
      file = strvcat(file,rep(line,m));
    end

    if (ind_inc)
      file = strvcat(file,...
         '  __shared__ int    nelems2, ncolor;');
    end

    file = strvcat(file,...
     '  __shared__ int    nelem, offset_b;',' ',...
     '  extern __shared__ char shared[];',' ',...
     '  if (blockIdx.x+blockIdx.y*gridDim.x >= nblocks) return;',...
     '  if (threadIdx.x==0) {',' ',...
     '    // get sizes and shift pointers and direct-mapped data',' ',...
     '    int blockId = blkmap[blockIdx.x + blockIdx.y*gridDim.x  + block_offset];',' ',...
     '    nelem    = nelems[blockId];',...
     '    offset_b = offset[blockId];',' ');

    if (ind_inc)
      file = strvcat(file,...
         '    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);',...
         '    ncolor   = ncolors[blockId];',' ');
    end

    for m = 1:ninds
      line = ['    ind_ARG_size = ind_arg_sizes[' ...
              int2str(m-1) '+blockId*' int2str(ninds) '];'];
      file = strvcat(file,rep(line,m));
    end
    file = strvcat(file,' ');
    for m = 1:ninds
      line = ['    ind_ARG_map = &ind_map[',int2str(cumulative_indirect_index(find(inds == m, 1))) ,'*set_size] + ind_arg_offs[' ...
              int2str(m-1) '+blockId*' int2str(ninds) '];'];
      file = strvcat(file,rep(line,m));
    end

    file = strvcat(file,' ','    // set shared memory pointers',' ',...
                            '    int nbytes = 0;');
    for m = 1:ninds
      line = '    ind_ARG_s = (INDTYP *) &shared[nbytes];';
      file = strvcat(file,rep(line,m));
      if (m<ninds)
        line = ...
        '    nbytes    += ROUND_UP(ind_ARG_size*sizeof(INDTYP)*INDDIM);';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,'  }',' ',...
      '  __syncthreads(); // make sure all of above completed',' ',...
      '  // copy indirect datasets into shared memory or zero increment',' ');
    for m = 1:ninds
      if(indaccs(m)==OP_READ || indaccs(m)==OP_RW || indaccs(m)==OP_INC)
        line = '  for (int n=threadIdx.x; n<INDARG_size*INDDIM; n+=blockDim.x)';
        file = strvcat(file,rep(line,m));
        if(indaccs(m)==OP_READ || indaccs(m)==OP_RW)
          line = '    INDARG_s[n] = INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM];';
        elseif(indaccs(m)==OP_INC)
          line = '    INDARG_s[n] = ZERO_INDTYP;';
        end
        file = strvcat(file,rep(line,m),' ');

      end
    end

    file = strvcat(file,'  __syncthreads();',' ',...
                        '  // process set elements',' ');

    if (ind_inc)
      file = strvcat(file,...
        '  for (int n=threadIdx.x; n<nelems2; n+=blockDim.x) {',...
        '    int col2 = -1;                               ',' ',...
        '    if (n<nelem) {                               ',' ',...
        '      // initialise local variables              ',' ');

      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = '      for (int d=0; d<DIM; d++)';
          file = strvcat(file,rep(line,m));
          line = '        ARG_l[d] = ZERO_TYP;';
          file = strvcat(file,rep(line,m));
        end
      end

    else
      file = strvcat(file,...
             '  for (int n=threadIdx.x; n<nelem; n+=blockDim.x) {');
    end

%
% simple alternative when no indirection
%
  else

    use_shared = 0;
    for m = 1:nargs
      if(maps(m)~=OP_GBL && ~strcmp(dims{m},'1') )
        use_shared = 1;
      end
    end

    if (use_shared)
      file = strvcat(file,...
        '  int   tid = threadIdx.x%OP_WARPSIZE;',' ',...
        '  extern __shared__ char shared[];    ',' ',...
        '  char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);');
    end

    file = strvcat(file,' ',...
      '  // process set elements',' ', ...
      '  for (int n=threadIdx.x+blockIdx.x*blockDim.x;', ...
      '       n<set_size; n+=blockDim.x*gridDim.x) {');

    if (use_shared)
      file = strvcat(file,' ',...
        '    int offset = n - tid;',...
        '    int nelems = MIN(OP_WARPSIZE,set_size-offset);',' ',...
        '    // copy data into shared memory, then into local',' ');
    end

    for m = 1:nargs
      if(maps(m)~=OP_GBL && accs(m)~=OP_WRITE && ~strcmp(dims{m},'1') && ~soaflags(m))
        line = '    for (int m=0; m<DIM; m++)';
        file = strvcat(file,rep(line,m));
        line = ['      ((TYP *)arg_s)[tid+m*nelems] =' ...
                            ' ARG[tid+m*nelems+offset*DIM];'];
        file = strvcat(file,rep(line,m),' ');
        line = '    for (int m=0; m<DIM; m++)';
        file = strvcat(file,rep(line,m));
        line = '      ARG_l[m] = ((TYP *)arg_s)[m+tid*DIM];';
        file = strvcat(file,rep(line,m),' ');
      end
    end

  end

%
% kernel call
%
  if (ninds>0)
    prefix = '      ';
  else
    prefix = '    ';
  end

  % xxx: array of pointers for non-locals
  for  m = 1:ninds
    if (sum(inds==m)>1)
      if (indaccs(m) ~= OP_INC)
        file = strvcat(file,' ');
        line = '';
        ctr = 0;
        for n = 1:nargs
          if (inds(n) == m  && vectorised(m))
            file = strvcat(file,line);
            line = [ prefix ...
              sprintf(['arg%d_vec[%d] = ind_arg%d_s+arg_map[',int2str(cumulative_indirect_index(n)),'*set_size+n+offset_b]*DIM;'],m-1, ctr, inds(n)-1) ];
            line = rep(line,n);
            ctr = ctr+1;
          end
        end
        file = strvcat(file,line);
      end
    end
  end

  file = strvcat(file,...
                 ' ',[ prefix '// user-supplied kernel call'],' ');

  out = '';
  for m = 1:nargs
    line = [prefix name '( '];
    if (m~=1)
      line = blanks(length(line));
    end
    if (maps(m)==OP_GBL)
      if (accs(m)==OP_READ)
        line = rep([ line ' ARG,' ],m);
      else
        line = rep([ line ' ARG_l,' ],m);
      end
    elseif (maps(m)==OP_MAP & accs(m)==OP_INC && vectorised(m)==0)
      line = rep([ line ' ARG_l,' ],m);
    elseif (maps(m)==OP_MAP && vectorised(m)==0)
      line = rep([ line sprintf([' ind_arg%d_s+arg_map[',int2str(cumulative_indirect_index(m)),'*set_size+n+offset_b]*DIM,'],inds(m)-1) ],m);
    elseif (maps(m)==OP_MAP && m == 1)
      line = rep([ line ' ARG_vec,' ], inds(m));
    elseif (maps(m)==OP_MAP && m>1 && vectorised(m) ~= vectorised(m-1)) %xxx:vector
      line = rep([ line ' ARG_vec,' ], inds(m));
    elseif (maps(m)==OP_MAP && m>1 && vectorised(m) == vectorised(m-1))
      line = '';
    elseif (maps(m)==OP_ID)
      if (ninds>0)
        if (soaflags(m))
            line = rep([ line ' ARG+(n+offset_b),' ],m);
        else
            line = rep([ line ' ARG+(n+offset_b)*DIM,' ],m);
        end
      else
        if (strcmp(dims{m},'1') || soaflags(m))
          line = rep([ line ' ARG+n,' ],m);
        else
          line = rep([ line ' ARG_l,' ],m);
        end
      end
    else
      error('internal error 1')
    end
    if (m==nargs) %xx one line, and
      if (isempty(line))
        out = [out(1:end-1) ');'];
      else
        out = sprintf('%s\n%s );',out ,line(1:end-1));
      end
    else
      if (~isempty(line))
        out =  sprintf('%s\n%s',out ,rep(line,m));
      end
    end
  end
  file = strvcat(file,out);

%
% updating for indirect kernels ...
%

  if(ninds>0)
    if(ind_inc)
      file = strvcat(file,...
             ' ','      col2 = colors[n+offset_b];        ',...
                 '    }                                   ',...
             ' ','    // store local variables            ',' ');

            for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = sprintf('      int ARG_map;');
          file = strvcat(file,rep(line,m));
        end
      end
      file = strvcat(file,...
                ' ','      if (col2>=0) {                  ');
      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = ['        ARG_map = arg_map[',int2str(cumulative_indirect_index(m)),'*set_size+n+offset_b];'];
          file = strvcat(file,rep(line,m));
        end
      end
      file = strvcat(file,...
                '      }                  ');

      file = strvcat(file,...
            ' ','    for (int col=0; col<ncolor; col++) {',...
                '      if (col2==col) {                  ');
      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = '        for (int d=0; d<DIM; d++)';
          file = strvcat(file,rep(line,m));
          line = sprintf('          ind_arg%d_s[d+ARG_map*DIM] += ARG_l[d];',inds(m)-1);
          file = strvcat(file,rep(line,m));
        end
      end
      file = strvcat(file,'      }','      __syncthreads();','    }',' ');
    end

    file = strvcat(file,'  }',' ');
    if(max(indaccs(1:ninds)~=OP_READ)>0)
      file = strvcat(file,'  // apply pointered write/increment',' ');
    end
    for m = 1:ninds
      if(indaccs(m)==OP_WRITE || indaccs(m)==OP_RW || indaccs(m)==OP_INC)
        line = '  for (int n=threadIdx.x; n<INDARG_size*INDDIM; n+=blockDim.x)';
        file = strvcat(file,rep(line,m));
        if(indaccs(m)==OP_WRITE || indaccs(m)==OP_RW)
          line = '    INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] = INDARG_s[n];';
          file = strvcat(file,rep(line,m),' ');
        elseif(indaccs(m)==OP_INC)
          line = '    INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] += INDARG_s[n];';
          file = strvcat(file,rep(line,m),' ');
        end
      end
    end
%
% ... and direct kernels
%
  else

    if (use_shared)
      file = strvcat(file,' ',...
         '    // copy back into shared memory, then to device',' ');
    end

    for m = 1:nargs
      if(maps(m)~=OP_GBL && accs(m)~=OP_READ && ~strcmp(dims{m},'1') && ~soaflags(m))
        line = '    for (int m=0; m<DIM; m++)';
        file = strvcat(file,rep(line,m));
        line = '      ((TYP *)arg_s)[m+tid*DIM] = ARG_l[m];';
        file = strvcat(file,rep(line,m),' ');
        line = '    for (int m=0; m<DIM; m++)';
        file = strvcat(file,rep(line,m));
        line = '      ARG[tid+m*nelems+offset*DIM] = ((TYP *)arg_s)[tid+m*nelems];';
        file = strvcat(file,rep(line,m),' ');
      end
    end

    file = strvcat(file,'  }');
  end

%
% global reduction
%
  if (reduct)
    file = strvcat(file,' ','  // global reductions',' ');
    for m = 1:nargs
      if (maps(m)==OP_GBL && accs(m)~=OP_READ)
        line = '  for(int d=0; d<DIM; d++)';
        file = strvcat(file,rep(line,m));
        if(accs(m)==OP_INC)
          line = '    op_reduction<OP_INC>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);';
        elseif (accs(m)==OP_MIN)
          line = '    op_reduction<OP_MIN>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);';
        elseif (accs(m)==OP_MAX)
          line = '    op_reduction<OP_MAX>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);';
        else
          error('internal error: invalid reduction option')
        end
        file = strvcat(file,rep(line,m));
      end
    end
  end

  file = strvcat(file,'}');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% then C++ stub function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  file = strvcat(file,' ',' ','// host stub function          ',' ',...
        ['void op_par_loop_' name '(char const *name, op_set set,']);

  for m = unique_args
    line = rep('  op_arg ARG', m);

    if (m == unique_args(end))
      file = strvcat(file,[line ' ){'],' ');
    else
      file = strvcat(file,[line ',']);
    end
  end

  for m = 1:nargs
    if (maps(m)==OP_GBL)
      line = '  TYP *ARGh = (TYP *)ARG.data;';
      file = strvcat(file,rep(line,m));
    end
  end

  file = strvcat(file,' ',['  int    nargs   = ' num2str(nargs) ';']);

  line = sprintf('  op_arg args[%d];',nargs);
  file = strvcat(file,rep(line,m),' ');

  for m = 1:nargs
      if (find(m==unique_args,1) & (vectorised(m) > 0))
          line = '  ARG.idx = 0;';
          line = sprintf('%s\n  args[%d] = ARG;',line, m-1);
          file = strvcat(file,rep(line,m));
          first = find(vectorised == vectorised(m), 1);
          line = sprintf('  for (int v = 1; v < %d; v++) {\n',sum(vectorised == vectorised(m)) );
          line = sprintf('%s    args[%d + v] = op_arg_dat(arg%d.dat, v, arg%d.map, DIM, "TYP", %s);\n  }', line,  m-1, first-1, first-1, accsstring{accs(m)});
          file = strvcat(file,rep(line,m));
      elseif (vectorised(m)>0)

      else
          line = sprintf('  args[%d] = ARG;',m-1);
          file = strvcat(file,rep(line,m));
      end
  end
%
%   for m = 1:nargs
%       line = [line 'arg' num2str(m-1) ','];
%   end
%   file = strvcat(file,[line(1:end-1) '};']);

%
%   indirect bits
%
  if (ninds>0)
    file = strvcat(file,' ',['  int    ninds   = ' num2str(ninds) ';']);

    line = sprintf('  int    inds[%d] = {',nargs);
    for m = 1:nargs
      line = strcat(line,num2str(inds(m)-1),',');
    end
    file = strvcat(file,[line(1:end-1) '};']);

    file = strvcat(file,' ',...
      '  if (OP_diags>2) {              ',...
     ['    printf(" kernel routine with indirection: ' name '\n");'],...
      '  }                              ',' ',...
      '  // get plan                    ',' ',...
     ['  #ifdef OP_PART_SIZE_'            num2str(nk-1)    ],...
     ['    int part_size = OP_PART_SIZE_' num2str(nk-1) ';'],...
      '  #else                          ',...
      '    int part_size = OP_part_size;',...
      '  #endif                         ',' ',...
      '  int set_size = op_mpi_halo_exchanges(set, nargs, args);                          ');

%
% direct bit
%
  else
    file = strvcat(file,' ',...
     '  if (OP_diags>2) {              ',...
    ['    printf(" kernel routine w/o indirection:  ' name '\n");'],...
     '  }                              ',' ',...
      '  op_mpi_halo_exchanges(set, nargs, args);                          ');
  end

%
% start timing
%
  file = strvcat(file,' ','  // initialise timers                    ',...
                      ' ','  double cpu_t1, cpu_t2, wall_t1, wall_t2;',...
                          '  op_timers_core(&cpu_t1, &wall_t1);           ');

  file = strvcat(file,' ','  if (set->size >0) {',' ');

  if (sum(soaflags))
      file = strvcat(file,'    int op2_stride = set->size + set->exec_size + set->nonexec_size;');
      file = strvcat(file,'    op_decl_const_char(1, "int", sizeof(int), (char *)&op2_stride, "op2_stride");',' ');
  end

  if (ninds>0)
      file = strvcat(file,'    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);',' ');
  end
%
% transfer constants
%
  if (length(find(maps(1:nargs)==OP_GBL & accs(1:nargs)==OP_READ))>0)
    file = strvcat(file,'  ',...
     '    // transfer constants to GPU',' ',...
     '    int consts_bytes = 0;');
    for m=1:nargs
      if(maps(m)==OP_GBL && accs(m)==OP_READ);
        line = '    consts_bytes += ROUND_UP(DIM*sizeof(TYP));';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,'  ',...
     '    reallocConstArrays(consts_bytes);',' ',...
     '    consts_bytes = 0;');

    for m=1:nargs
      if(maps(m)==OP_GBL && accs(m)==OP_READ);
        line = '    ARG.data   = OP_consts_h + consts_bytes;';
        file = strvcat(file,rep(line,m));
        line = '    ARG.data_d = OP_consts_d + consts_bytes;';
        file = strvcat(file,rep(line,m));
        line = ...
        '    for (int d=0; d<DIM; d++) ((TYP *)ARG.data)[d] = ARGh[d];';
        file = strvcat(file,rep(line,m));
        line = '    consts_bytes += ROUND_UP(DIM*sizeof(TYP));';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,'  ','    mvConstArraysToDevice(consts_bytes);');
  end

%
% transfer global reduction initial data
%
  if (ninds==0)
    file = strvcat(file,' ',...
      '    // set CUDA execution parameters  ',' ',...
     ['    #ifdef OP_BLOCK_SIZE_'          num2str(nk-1)    ],...
     ['      int nthread = OP_BLOCK_SIZE_' num2str(nk-1) ';'],...
      '    #else                             ',...
      '      // int nthread = OP_block_size; ',...
      '      int nthread = 128;              ',...
      '    #endif                            ',' ',...
      '    int nblocks = 200;                ');
  end

  if (reduct)
    file = strvcat(file,'  ',...
         '    // transfer global reduction data to GPU',' ');

    if (ninds>0)
      file = strvcat(file,...
         '    int maxblocks = 0;',...
         '    for (int col=0; col < Plan->ncolors; col++)',...
         '      maxblocks = MAX(maxblocks,Plan->ncolblk[col]);');
    else
      file = strvcat(file,'    int maxblocks = nblocks;');
    end

    file = strvcat(file,' ','    int reduct_bytes = 0;',...
                            '    int reduct_size  = 0;');

    for m=1:nargs
      if(maps(m)==OP_GBL && accs(m)~=OP_READ);
        line = '    reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));';
        file = strvcat(file,rep(line,m));
        line = '    reduct_size   = MAX(reduct_size,sizeof(TYP));';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,'  ',...
     '    reallocReductArrays(reduct_bytes);',' ',...
     '    reduct_bytes = 0;');

    for m=1:nargs
      if(maps(m)==OP_GBL && accs(m)~=OP_READ);
        line = '    ARG.data   = OP_reduct_h + reduct_bytes;';
        file = strvcat(file,rep(line,m));
        line = '    ARG.data_d = OP_reduct_d + reduct_bytes;';
        file = strvcat(file,rep(line,m));
        file = strvcat(file,'    for (int b=0; b<maxblocks; b++)');
        line = '      for (int d=0; d<DIM; d++)';
        file = strvcat(file,rep(line,m));
        if (accs(m)==OP_INC)
          line = '        ((TYP *)ARG.data)[d+b*DIM] = ZERO_TYP;';
        else
          line = '        ((TYP *)ARG.data)[d+b*DIM] = ARGh[d];';
        end
        file = strvcat(file,rep(line,m));
        line = '    reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,'  ','    mvReductArraysToDevice(reduct_bytes);');
  end

%
% kernel call for indirect version
%
  if (ninds>0)
    file = strvcat(file,' ',...
      '    // execute plan                                            ',' ',...
      '    int block_offset = 0;                                      ',' ',...
      '    for (int col=0; col < Plan->ncolors; col++) {              ',' ',...
      '      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs,args);',' ',...
     ['    #ifdef OP_BLOCK_SIZE_'          num2str(nk-1)               ],...
     ['      int nthread = OP_BLOCK_SIZE_' num2str(nk-1) ';           '],...
      '    #else                                                      ',...
      '      int nthread = OP_block_size;                             ',...
      '    #endif                                                     ',' ',...
      '      dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],',...
      '                      Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);',...
      '      if (Plan->ncolblk[col] > 0) {');

    if (reduct)
      file = strvcat(file,...
        '        int nshared = MAX(Plan->nshared,reduct_size*nthread);');
    else
      file = strvcat(file,'        int nshared = Plan->nsharedCol[col];');
    end

      file = strvcat(file,...
       ['        op_cuda_' name '<<<nblocks,nthread,nshared>>>(']);

      for m = 1:ninds
        line = sprintf('           (TYP *)ARG.data_d,',m-1);
        file = strvcat(file,rep(line,invinds(m)));
      end

      file = strvcat(file,'           Plan->ind_map,');
      file = strvcat(file,'           Plan->loc_map,');

      for m = 1:nargs
        if (inds(m)==0)
          line = '           (TYP *)ARG.data_d,';
          file = strvcat(file,rep(line,m));
        end
      end

      file = strvcat(file, ...
       '           Plan->ind_sizes,                                     ',...
       '           Plan->ind_offs,                                      ',...
       '           block_offset,                                        ',...
       '           Plan->blkmap,                                        ',...
       '           Plan->offset,                                        ',...
       '           Plan->nelems,                                        ',...
       '           Plan->nthrcol,                                       ',...
       '           Plan->thrcol,                                        ',...
       '           Plan->ncolblk[col],                                  ',...
       '           set_size);                                           ',...
       ' ','        cutilSafeCall(cudaThreadSynchronize());             ',...
      ['        cutilCheckMsg("op_cuda_' name ' execution failed\n");']);
      if (reduct)
        file = strvcat(file, ...
        ' ','        // transfer global reduction data back to CPU',...
        ' ','        if (col == Plan->ncolors_owned)',...
        ' ','          mvReductArraysToHost(reduct_bytes);',' ');
      end
      file = strvcat(file, ...
       '      }                                                         ',...
       ' ','      block_offset += Plan->ncolblk[col];                   ',...
       '    }                                                         ');

%
% kernel call for direct version
%
  else
    file = strvcat(file,...
      ' ','    // work out shared memory requirements per element',...
      ' ','    int nshared = 0;');

    for m = 1:nargs
      if(maps(m)~=OP_GBL && ~strcmp(dims{m},'1'));
        line = '    nshared = MAX(nshared,sizeof(TYP)*DIM);';
        file = strvcat(file,rep(line,m));
      end
    end

    file = strvcat(file,...
      ' ','    // execute plan                    ',' ',...
          '    int offset_s = nshared*OP_WARPSIZE;',' ');

    if (reduct)
      file = strvcat(file,...
       '    nshared = MAX(nshared*nthread,reduct_size*nthread);',' ');
    else
      file = strvcat(file,'    nshared = nshared*nthread;',' ');
    end
    line = ['    op_cuda_' name '<<<nblocks,nthread,nshared>>>( '];

    for m = 1:nargs
      file = strvcat(file,rep([line '(TYP *) ARG.data_d,'],m));
      line = blanks(length(line));
    end

    file = strvcat(file,[ line 'offset_s,'  ],...
                        [ line 'set->size );'],' ',...
      '    cutilSafeCall(cudaThreadSynchronize());                ', ...
     ['    cutilCheckMsg("op_cuda_', name ' execution failed\n");']);
  end

   if (ninds>0)
   file = strvcat(file,' ',['    op_timing_realloc(' num2str(nk-1) ');                       ']);
   file = strvcat(file,...
    ['    OP_kernels[' num2str(nk-1) '].transfer  += Plan->transfer; '],...
    ['    OP_kernels[' num2str(nk-1) '].transfer2 += Plan->transfer2;']);
   end

%
% transfer global reduction initial data
%
  if (reduct)
    if (ninds == 0)
      file = strvcat(file,...
           ' ','    // transfer global reduction data back to CPU',...
           ' ','    mvReductArraysToHost(reduct_bytes);',' ');
    end
    for m=1:nargs
      if(maps(m)==OP_GBL && accs(m)~=OP_READ);
        file = strvcat(file,'    for (int b=0; b<maxblocks; b++)');
        line = '      for (int d=0; d<DIM; d++)';
        file = strvcat(file,rep(line,m));
        if (accs(m)==OP_INC)
          line = '        ARGh[d] = ARGh[d] + ((TYP *)ARG.data)[d+b*DIM];';
        elseif (accs(m)==OP_MIN)
          line = '        ARGh[d] = MIN(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);';
        elseif (accs(m)==OP_MAX)
          line = '        ARGh[d] = MAX(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);';
        end
        file = strvcat(file,rep(line,m));
        line = '  ARG.data = (char *)ARGh;';
        file = strvcat(file,' ',rep(line,m));
        line = '  op_mpi_reduce(&ARG,ARGh);';
        file = strvcat(file,' ',rep(line,m));
      end
    end
  end
  file = strvcat(file, ' ','  }',' ');

  file = strvcat(file,' ','  op_mpi_set_dirtybit(nargs, args);');

%
% update kernel record
%

  file = strvcat(file,' ','  // update kernel record',' ',...
     '  op_timers_core(&cpu_t2, &wall_t2);                               ',...
    ['  op_timing_realloc(' num2str(nk-1) ');                       '],...
    ['  OP_kernels[' num2str(nk-1) '].name      = name;             '],...
    ['  OP_kernels[' num2str(nk-1) '].count    += 1;                '],...
    ['  OP_kernels[' num2str(nk-1) '].time     += wall_t2 - wall_t1;']);
  if (ninds>0)
   %file = strvcat(file,...
    %['  OP_kernels[' num2str(nk-1) '].transfer  += Plan->transfer; '],...
    %['  OP_kernels[' num2str(nk-1) '].transfer2 += Plan->transfer2;']);
  else

   line = ...
    ['  OP_kernels[' num2str(nk-1) '].transfer += (float)set->size *'];

   for m = 1:nargs
     if(maps(m)~=OP_GBL)
       if (accs(m)==OP_READ || accs(m)==OP_WRITE)
         file = strvcat(file,rep([line ' ARG.size;'],m));
       else
         file = strvcat(file,rep([line ' ARG.size * 2.0f;'],m));
       end
     end
   end
  end

  file = strvcat(file,'} ',' ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  output individual kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  fid = fopen([name '_kernel.cu'],'wt');

  fprintf(fid,'//\n// auto-generated by op2.m on %s\n//\n\n',date);
  for n=1:size(file,1)
    fprintf(fid,'%s\n',deblank(file(n,:)));
  end
  fclose(fid);

end  % end of main kernel call loop


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  output one master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% change to this below for new version
%     '#include "op_lib_cpp.h"          ',...
%     '#include "op_cuda_rt_support.h"',' ',...

file = strvcat('// header                 ',' ',...
               '#include "op_lib_cpp.h"      ',' ',...
               '#include "op_cuda_rt_support.h"      ',...
               '#include "op_cuda_reduction.h" ',...
               '// global constants       ',' ',...
               '#ifndef MAX_CONST_SIZE    ',...
               '#define MAX_CONST_SIZE 128',...
               '#endif                    ',' ');

for nc = 1:length(consts)
  if (consts{nc}.dim==1)
    file = strvcat(file, ...
      [ '__constant__ ' consts{nc}.type ' ' consts{nc}.name ';' ]);
  else
    if (consts{nc}.dim>0)
      num = num2str(consts{nc}.dim);
    else
      num = 'MAX_CONST_SIZE';
    end
    file = strvcat(file, ...
      [ '__constant__ ' consts{nc}.type ' ' consts{nc}.name '[' num '];' ]);
  end
end

if (any_soa)
   file = strvcat(file, '__constant__ int op2_stride;', ' ',...
       '#define OP2_STRIDE(arr, idx) arr[op2_stride*(idx)]');
   %file = strvcat(file, 'extern int op2_stride;');
end

file = strvcat(file,' ',...
      'void op_decl_const_char(int dim, char const *type,',...
      '            int size, char *dat, char const *name){');
for nc = 1:length(consts)
  if (consts{nc}.dim<0)
    file = strvcat(file,...
     ['  if(~strcmp(name,"' name '") && size>MAX_CONST_SIZE) {'],...
     ['    printf("error: MAX_CONST_SIZE not big enough\n"); exit(1);'],...
      '  }');
  end
end
file = strvcat(file,...
      '  cutilSafeCall(cudaMemcpyToSymbol(name, dat, dim*size));',...
      '} ',' ',...
      '// user kernel files',' ');

for nk = 1:length(kernels)
  file = strvcat(file,...
     ['#include "' kernels{nk}.name '_kernel.cu"']);
end

fid = fopen([ master '_kernels.cu'],'wt');

fprintf(fid,'//\n// auto-generated by op2.m on %s\n//\n\n',date);

for n=1:size(file,1)
  fprintf(fid,'%s\n',deblank(file(n,:)));
end

fclose(fid);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% a little function to replace keywords
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function line = rep(line,m)

global dims idxs typs indtyps inddims

if (m <= length(inddims))
  line = regexprep(line,'INDDIM',char(inddims{m}));
  line = regexprep(line,'INDTYP',char(indtyps{m}));
end
line = regexprep(line,'INDARG',sprintf('ind_arg%d',m-1));

line = regexprep(line,'DIM',dims(m));
line = regexprep(line,'ARG',sprintf('arg%d',m-1));
line = regexprep(line,'TYP',typs(m));
line = regexprep(line,'IDX',num2str(idxs(m)));

end
