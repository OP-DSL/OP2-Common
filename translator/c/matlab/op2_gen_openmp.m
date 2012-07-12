%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% OpenMP code generator
%
% This routine is called by op2 which parses the input files
%
% It produces a file xxx_kernel.cpp for each kernel,
% plus a master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function op2_gen_openmp(master,date,consts,kernels)

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
                  ['#include "' name '.h"'],' ',' ',...
                   '// x86 kernel function',' ',...
                  ['void op_x86_' name '(']);
  if (ninds>0)
    file = strvcat(file,'  int    blockIdx,');
  end

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
                        '  int   set_size) {    ',' ');
  else
    file = strvcat(file,'  int   start,    ',...
                        '  int   finish ) {',' ');
  end

  for m = 1:nargs
    if (maps(m)==OP_MAP && accs(m)==OP_INC)
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
      line = '  int   *ind_ARG_map, ind_ARG_size;';
      file = strvcat(file,rep(line,m));
    end
    for m = 1:ninds
      line = '  INDTYP *ind_ARG_s;';
      file = strvcat(file,rep(line,m));
    end

    file = strvcat(file,...
     '  int    nelem, offset_b;',' ',...
     '  char shared[128000];',' ',...
     '  if (0==0) {',' ',...
     '    // get sizes and shift pointers and direct-mapped data',' ',...
     '    int blockId = blkmap[blockIdx + block_offset];',...
     '    nelem    = nelems[blockId];',...
     '    offset_b = offset[blockId];',' ');

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
     '  // copy indirect datasets into shared memory or zero increment',' ');
    for m = 1:ninds
      if(indaccs(m)==OP_READ || indaccs(m)==OP_RW || indaccs(m)==OP_INC)
        line = '  for (int n=0; n<INDARG_size; n++)';
        file = strvcat(file,rep(line,m));
        line = '    for (int d=0; d<INDDIM; d++)';
        file = strvcat(file,rep(line,m));
        if(indaccs(m)==OP_READ || indaccs(m)==OP_RW)
          line = '      INDARG_s[d+n*INDDIM] = INDARG[d+INDARG_map[n]*INDDIM];';
        elseif(indaccs(m)==OP_INC)
          line = '      INDARG_s[d+n*INDDIM] = ZERO_INDTYP;';
        end
        file = strvcat(file,rep(line,m),' ');
      end
    end

    file = strvcat(file,' ','  // process set elements',' ');

    if (ind_inc)
      file = strvcat(file,...
           '  for (int n=0; n<nelem; n++) {',' ',...
           '    // initialise local variables            ',' ');

      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = '    for (int d=0; d<DIM; d++)';
          file = strvcat(file,rep(line,m));
          line = '      ARG_l[d] = ZERO_TYP;';
          file = strvcat(file,rep(line,m));
        end
      end

    else
      file = strvcat(file,...
             '  for (int n=0; n<nelem; n++) {');
    end

%
% simple alternative when no indirection
%
  else

    file = strvcat(file,' ','  // process set elements',' ', ...
                        '  for (int n=start; n<finish; n++) {');
  end

%
% kernel call
%
  if (ninds>0)
    prefix = '    ';
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
      line = rep([ line ' ARG,' ],m);
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
        line = rep([ line ' ARG+(n+offset_b)*DIM,' ],m);
      else
        line = rep([ line ' ARG+n*DIM,' ],m);
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
             ' ','    // store local variables            ',' ');

      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = ['    int ARG_map = arg_map[',int2str(cumulative_indirect_index(m)),'*set_size+n+offset_b];'];
          file = strvcat(file,rep(line,m));
        end
      end

      for m = 1:nargs
        if (maps(m)==OP_MAP && accs(m)==OP_INC)
          line = '    for (int d=0; d<DIM; d++)';
          file = strvcat(file,' ',rep(line,m));
          line = sprintf('      ind_arg%d_s[d+ARG_map*DIM] += ARG_l[d];',inds(m)-1);
          file = strvcat(file,rep(line,m));
        end
      end
    end

    file = strvcat(file,'  }',' ');
    if(max(indaccs(1:ninds)~=OP_READ)>0)
      file = strvcat(file,'  // apply pointered write/increment',' ');
    end
    for m = 1:ninds
      if(indaccs(m)==OP_WRITE || indaccs(m)==OP_RW || indaccs(m)==OP_INC)
        line = '  for (int n=0; n<INDARG_size; n++)';
        file = strvcat(file,rep(line,m));
        line = '    for (int d=0; d<INDDIM; d++)';
        file = strvcat(file,rep(line,m));
        if(indaccs(m)==OP_WRITE || indaccs(m)==OP_RW)
          line = '      INDARG[d+INDARG_map[n]*INDDIM] = INDARG_s[d+n*INDDIM];';
          file = strvcat(file,rep(line,m),' ');
        elseif(indaccs(m)==OP_INC)
          line = '      INDARG[d+INDARG_map[n]*INDDIM] += INDARG_s[d+n*INDDIM];';
          file = strvcat(file,rep(line,m),' ');
        end
      end
    end
%
% ... and direct kernels
%
  else

    file = strvcat(file,'  }');
  end

%
% global reduction
%

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
    if (maps(m)==OP_GBL && accs(m)~=OP_READ)
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


%
% set number of threads in x86 execution and create arrays for reduction
%

  if (reduct || ninds==0)
    file = strvcat(file,' ','  // set number of threads          ',' ',...
                            '#ifdef _OPENMP                          ',...
                            '  int nthreads = omp_get_max_threads( );',...
                            '#else                                   ',...
                            '  int nthreads = 1;                     ',...
                            '#endif                                  ');
  end

  if (reduct)
    file = strvcat(file,' ',...
           '  // allocate and initialise arrays for global reduction');
    for m = 1:nargs
      if (maps(m)==OP_GBL && accs(m)~=OP_READ)
        line = '  TYP ARG_l[DIM+64*64];';
        file = strvcat(file,' ',rep(line,m),...
                            '  for (int thr=0; thr<nthreads; thr++)');
        if (accs(m)==OP_INC)
          line = '    for (int d=0; d<DIM; d++) ARG_l[d+thr*64]=ZERO_TYP;';
        else
          line = '    for (int d=0; d<DIM; d++) ARG_l[d+thr*64]=ARGh[d];';
        end
        file = strvcat(file,rep(line,m));
      end
    end
  end

  file = strvcat(file,' ','  if (set->size >0) {',' ');

%
% kernel call for indirect version
%
  if (ninds>0)

  file = strvcat(file,' ',...
      '    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);',...
      '    // execute plan                                            ',' ',...
      '    int block_offset = 0;                                      ',' ',...
      '    for (int col=0; col < Plan->ncolors; col++) {              ',...
      '      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs, args);',' ',...
      '      int nblocks = Plan->ncolblk[col];                        ',' ',...
      '#pragma omp parallel for                                     ',...
      '      for (int blockIdx=0; blockIdx<nblocks; blockIdx++)       ',...
     ['      op_x86_' name '( blockIdx,                              '] );

  for m = 1:ninds
    line = sprintf('         (TYP *)ARG.data,');
    file = strvcat(file,rep(line,invinds(m)));
  end

  file = strvcat(file,'         Plan->ind_map,');
  file = strvcat(file,'         Plan->loc_map,');

  for m = 1:nargs
    if (inds(m)==0)
      line = '         (TYP *)ARG.data,';
      file = strvcat(file,rep(line,m));
    end
  end

  file = strvcat(file, ...
    '         Plan->ind_sizes,                              ',...
    '         Plan->ind_offs,                               ',...
    '         block_offset,                                 ',...
    '         Plan->blkmap,                                 ',...
    '         Plan->offset,                                 ',...
    '         Plan->nelems,                                 ',...
    '         Plan->nthrcol,                                ',...
    '         Plan->thrcol,                                 ',...
    '         set_size);                                    ',' ',...
    '      block_offset += nblocks;                         ',...
    '    }                                                  ');

%
% kernel call for direct version
%
  else
    file = strvcat(file,...
       ' ','  // execute plan                            ',...
       ' ','#pragma omp parallel for                     ',...
           '  for (int thr=0; thr<nthreads; thr++) {     ',...
           '    int start  = (set->size* thr   )/nthreads;',...
           '    int finish = (set->size*(thr+1))/nthreads;');
    line = ['    op_x86_' name '( '];

    for m = 1:nargs
      if(maps(m)==OP_GBL && accs(m)~=OP_READ);
        file = strvcat(file,rep([line 'ARG_l + thr*64,'],m));
      else
        file = strvcat(file,rep([line '(TYP *) ARG.data,'],m));
      end
      line = blanks(length(line));
    end

    file = strvcat(file,[ line 'start, finish );'],'  }');
  end
  if (ninds>0)
   file = strvcat(file,' ',['  op_timing_realloc(' num2str(nk-1) ');                       ']);
   file = strvcat(file,...
    ['  OP_kernels[' num2str(nk-1) '].transfer  += Plan->transfer; '],...
    ['  OP_kernels[' num2str(nk-1) '].transfer2 += Plan->transfer2;']);
  end
  file = strvcat(file, ' ','  }',' ');
%
% combine reduction data from multiple OpenMP threads
%
  file = strvcat(file,' ','  // combine reduction data');
  for m=1:nargs
    if(maps(m)==OP_GBL && accs(m)~=OP_READ);
      file = strvcat(file,' ','  for (int thr=0; thr<nthreads; thr++)');
      if(accs(m)==OP_INC)
        line = '    for(int d=0; d<DIM; d++) ARGh[d] += ARG_l[d+thr*64];';
      elseif (accs(m)==OP_MIN)
        line = ...
   '    for(int d=0; d<DIM; d++) ARGh[d]  = MIN(ARGh[d],ARG_l[d+thr*64]);';
      elseif (accs(m)==OP_MAX)
        line = ...
   '    for(int d=0; d<DIM; d++) ARGh[d]  = MAX(ARGh[d],ARG_l[d+thr*64]);';
      else
        error('internal error: invalid reduction option')
      end
      file = strvcat(file,rep(line,m));
      line = '  op_mpi_reduce(&ARG,ARGh);';
      file = strvcat(file,' ',rep(line,m));
    end
  end

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

  if (ninds == 0)
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

  fid = fopen([name '_kernel.cpp'],'wt');

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
%     '#include "op_openmp_rt_support.h"',' ',...

file = strvcat('// header                 ',' ',...
               '#include "op_lib_cpp.h"       ',' ',...
               '// global constants       ',' ');

for nc = 1:length(consts)
  if (consts{nc}.dim==1)
    file = strvcat(file, ...
      [ 'extern ' consts{nc}.type ' ' consts{nc}.name ';' ]);
  else
    if (consts{nc}.dim>0)
      num = num2str(consts{nc}.dim);
    else
      num = 'MAX_CONST_SIZE';
    end
    file = strvcat(file, ...
      [ 'extern ' consts{nc}.type ' ' consts{nc}.name '[' num '];' ]);
  end
end

if (any_soa)
   file = strvcat(file, ' ','extern int op2_stride;','#define OP2_STRIDE(arr, idx) arr[idx]',' ');
end

file = strvcat(file,' ','// user kernel files',' ');

for nk = 1:length(kernels)
  file = strvcat(file,...
     ['#include "' kernels{nk}.name '_kernel.cpp"']);
end

fid = fopen([ master '_kernels.cpp'],'wt');

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

