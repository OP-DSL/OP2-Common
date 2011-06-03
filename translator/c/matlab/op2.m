%
% Source code transformation tool
%
% This tool parses the user's original source code
% to produce the CUDA code which will execute the
% user's kernel functions.
%
% The main deficiency of this current implementation
% is that it does not handle cases in which multiple 
% arguments involve the same underlying dataset.
%
% This prototype is written in MATLAB but a future
% version may use Python to avoid licensing costs.
% Alternatively, the MATLAB processor can be 
% "compiled" to produce a standalone version which
% can be freely distributed.
%
%
% usage: op2('filename')
%
% This takes as input 
%
% filename.cpp
%
% and produces as output
%
% filename_op.cpp
% filename_kernels.cu
%
% and one or more files of the form
%
% xxx_kernel.cu
%
% where xxx corresponds to the name of one of the
% kernel functions in filename.cpp
%

function op2(varargin)

global dims idxs typs indtyps inddims

%
% declare constants
%

OP_CUDA = 1;
OP_x86  = 2;

OP_targets = [ OP_CUDA OP_x86 ];

OP_ID  = 1;
OP_GBL = 2;
OP_MAP = 3;

OP_READ  = 1;
OP_WRITE = 2;
OP_RW    = 3;
OP_INC   = 4;
OP_MAX   = 5;
OP_MIN   = 6;

OP_accs_labels = { 'OP_READ' 'OP_WRITE' 'OP_RW' ...
                   'OP_INC'  'OP_MAX'   'OP_MIN' };

date = datestr(now);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  outer loop over all backend targets
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for target = OP_targets

disp(' ')
if (target==OP_CUDA)
  disp('processing files for CUDA target')
else
  disp('processing files for x86 target')
end

nconsts = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  loop over all input source files
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nker = -1;

for narg = 1: nargin
  nkernels = 0;

  filename = varargin{narg};
  disp(sprintf('\n processing file %d of %d (%s)',...
               narg,nargin,[filename '.cpp']));

  src_file = fileread([filename '.cpp']);

  consts    = op_decl_const_parse(src_file);
  n_consts  = length(consts);
  loc_consts = [];
  for n = 1:n_consts
    loc_consts(n) = consts{n}.loc;
  end

  loop_args  = op_par_loop_parse(src_file);
  n_loops    = length(loop_args);
  loc_loops  = [];
  for n = 1:n_loops
    loc_loops(n) = loop_args{n}.loc;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  process op_par_loop calls
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  for loop_index = 1:n_loops
    nkernels = nkernels + 1;
    nker     = nker + 1;

    fn_name = loop_args{loop_index}.name1;
    nargs   = loop_args{loop_index}.nargs;
    disp(sprintf('\n  processing kernel %d (%s) with %d arguments',...
                 nkernels,fn_name,nargs));

%
% process arguments
%

    vars = {};
    idxs = zeros(1,nargs);
    dim  = zeros(1,nargs);
    dims = {};
    maps = zeros(1,nargs);
    typs = {};
    accs = zeros(1,nargs);

    for m = 1:nargs
      type = loop_args{loop_index}.type{m};
      args = args_parse(loop_args{loop_index}.args{m});

      if (strcmp(type,'op_arg_dat'))
        vars{m} = args{1};
        idxs(m) = str2num(args{2});

	if (strcmp(args{3},'OP_ID'))
          maps(m) = OP_ID;
          if(idxs(m)~=-1)
            error(sprintf('invalid index for argument %d',m));
          end
        else
          maps(m) = OP_MAP;
          if(idxs(m)<0)
            error(sprintf('invalid index for argument %d',m));
          end
        end

        dims{m} = args{4};
        typs{m} = args{5}(2:end-1);

        if(isempty(strmatch(args{6},OP_accs_labels)))
          error(sprintf('unknown access type for argument %d',m));
        else
          accs(m) = strmatch(args{6},OP_accs_labels);
        end
      end

      if (strcmp(type,'op_arg_gbl'))
        maps(m) = OP_GBL;
        vars{m} = args{1};
        dims{m} = args{2};
        typs{m} = args{3}(2:end-1);

        if(isempty(strmatch(args{4},OP_accs_labels)))
          error(sprintf('unknown access type for argument %d',m));
        else
          accs(m) = strmatch(args{4},OP_accs_labels);
        end
      end

      if(maps(m)==OP_GBL & (accs(m)==OP_WRITE | accs(m)==OP_RW))
        error(sprintf('invalid access type for argument %d',m));
      end

      if(maps(m)~=OP_GBL & (accs(m)==OP_MIN | accs(m)==OP_MAX))
        error(sprintf('invalid access type for argument %d',m));
      end

    end

%
% set two logicals 
%

    ind_inc = max(maps==OP_MAP & accs==OP_INC)  > 0;
    reduct  = max(maps==OP_GBL & accs~=OP_READ) > 0;

%
%  identify indirect datasets
%

    ninds     = 0;
    invinds   = zeros(1,nargs);
    inds      = zeros(1,nargs);
    indtyps   = cell(1,nargs);
    inddims   = cell(1,nargs);
    indaccs   = zeros(1,nargs);

    j = find(maps==OP_MAP);                % find all indirect arguments

    while (~isempty(j))
      match = strcmp(vars(j(1)),   vars(j)) ...  % same variable name
            & strcmp(typs(j(1)),   typs(j)) ...  % same type  
            &       (accs(j(1)) == accs(j));     % same access
      ninds = ninds + 1;
      indtyps{ninds} = typs{j(1)};
      inddims{ninds} = dims{j(1)};
      indaccs(ninds) = accs(j(1));
      inds(j(find(match))) = ninds;
      invinds(ninds) = j(1);
      j = j(find(~match));            % find remaining indirect arguments
    end

%
% output various diagnostics
%

    disp(['    local constants:    ' ...
          num2str(find(maps==OP_GBL & accs==OP_READ)-1) ]);
    disp(['    global reductions:  ' ...
          num2str(find(maps==OP_GBL & accs~=OP_READ)-1) ]);
    disp(['    direct arguments:   ' num2str(find(maps==OP_ID)-1) ]);
    disp(['    indirect arguments: ' num2str(find(maps==OP_MAP)-1) ]);
    if (ninds>0)
      disp(['    number of indirect datasets: ' num2str(ninds) ]);
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  create new kernel file, starting with CUDA kernel function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (target==OP_CUDA)
      file = strvcat('// user function         ',' ',...
                     '__device__               ',...
                    ['#include "' fn_name '.h"'],' ',' ',...
                     '// CUDA kernel function',' ',...
                    ['__global__ void op_cuda_' fn_name '(']);

    elseif (target==OP_x86)
      file = strvcat('// user function         ',' ',...
                    ['#include "' fn_name '.h"'],' ',' ',...
                     '// x86 kernel function',' ',...
                    ['void op_x86_' fn_name '(']);
      if (ninds>0)
        file = strvcat(file,'  int    blockIdx,');
      end
    end

    for m = 1:ninds
      line = '  INDTYP *ind_ARG, int *ind_ARG_maps,';
      file = strvcat(file,rep(line,m));
    end

    for m = 1:nargs
      if (maps(m)==OP_GBL & accs(m)==OP_READ)
        line = '  const TYP *ARG,';    % declared const for performance
      elseif (maps(m)==OP_ID & ninds>0)
        line = '  TYP *ARG,';
      elseif (maps(m)==OP_GBL | maps(m)==OP_ID)
        line = '  TYP *ARG,';
      else
        line = '  short *ARG_maps,';
      end
      file = strvcat(file,rep(line,m));
    end

    if (ninds>0)
      file = strvcat(file,'  int   *ind_arg_sizes,',...
                          '  int   *ind_arg_offs,',...
                          '  int    block_offset,',...
                          '  int   *blkmap,      ',...
                          '  int   *offset,      ',...
                          '  int   *nelems,      ',...
                          '  int   *ncolors,     ',...
                          '  int   *colors) {    ',' ');
    else
      if (target==OP_CUDA)
        file = strvcat(file,'  int   offset_s,   ',...
                            '  int   set_size ) {',' ');
      elseif (target==OP_x86)
        file = strvcat(file,'  int   start,    ',...
                            '  int   finish ) {',' ');
      end
    end

    if (target==OP_CUDA)
      for m = 1:nargs
        if (maps(m)==OP_GBL & accs(m)~=OP_READ)
          line = '  TYP ARG_l[DIM];';
          file = strvcat(file,rep(line,m));
          if (accs(m)==OP_INC)
            line = '  for (int d=0; d<DIM; d++) ARG_l[d]=ZERO_TYP;';
          else
            line = ...
            '  for (int d=0; d<DIM; d++) ARG_l[d]=ARG[d+blockIdx.x*DIM];';
          end
          file = strvcat(file,rep(line,m));
        elseif (maps(m)==OP_MAP & accs(m)==OP_INC)
          line = '  TYP ARG_l[DIM];';
          file = strvcat(file,rep(line,m));
        elseif (ninds==0 & maps(m)==OP_ID & dims{m}~='1')
          line = '  TYP ARG_l[DIM];';
          file = strvcat(file,rep(line,m));
        end
      end

    elseif (target==OP_x86)
      for m = 1:nargs
        if (maps(m)==OP_MAP & accs(m)==OP_INC)
          line = '  TYP ARG_l[DIM];';
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

      if (target==OP_CUDA)
       file = strvcat(file,...
        '  __shared__ int    nelem, offset_b;',' ',...
        '  extern __shared__ char shared[];',' ',...
        '  if (threadIdx.x==0) {',' ',...
        '    // get sizes and shift pointers and direct-mapped data',' ',...
        '    int blockId = blkmap[blockIdx.x + block_offset];',' ',...
        '    nelem    = nelems[blockId];',...
        '    offset_b = offset[blockId];',' ');
      elseif (target==OP_x86)
       file = strvcat(file,...
        '  __shared__ int    nelem, offset_b;',' ',...
        '  __shared__ char shared[64000];',' ',...
        '  if (0==0) {',' ',...
        '    // get sizes and shift pointers and direct-mapped data',' ',...
        '    int blockId = blkmap[blockIdx + block_offset];',...
        '    nelem    = nelems[blockId];',...
        '    offset_b = offset[blockId];',' ');
      end

      if (ind_inc) 
        if (target==OP_CUDA)
        file = strvcat(file,...
           '    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);',...
           '    ncolor   = ncolors[blockId];',' ');
        elseif (target==OP_x86)
        file = strvcat(file,...
           '    nelems2  = nelem;',...
           '    ncolor   = ncolors[blockId];',' ');
        end
      end

      for m = 1:ninds
        line = ['    ind_ARG_size = ind_arg_sizes[' ...
                int2str(m-1) '+blockId*' int2str(ninds) '];'];
        file = strvcat(file,rep(line,m));
      end
      file = strvcat(file,' ');
      for m = 1:ninds
        line = ['    ind_ARG_map = ind_ARG_maps + ind_arg_offs[' ...
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
        if(indaccs(m)==OP_READ | indaccs(m)==OP_RW | indaccs(m)==OP_INC)
          if (target==OP_CUDA)
            line = '  for (int n=threadIdx.x; n<INDARG_size*INDDIM; n+=blockDim.x)';
            file = strvcat(file,rep(line,m));
            if(indaccs(m)==OP_READ | indaccs(m)==OP_RW)
              line = '    INDARG_s[n] = INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM];';
            elseif(indaccs(m)==OP_INC)
              line = '    INDARG_s[n] = ZERO_INDTYP;';
            end
            file = strvcat(file,rep(line,m),' ');

          elseif (target==OP_x86)
            line = '  for (int n=0; n<INDARG_size; n++)';
            file = strvcat(file,rep(line,m));
            line = '    for (int d=0; d<INDDIM; d++)';
            file = strvcat(file,rep(line,m));
            if(indaccs(m)==OP_READ | indaccs(m)==OP_RW)
              line = '      INDARG_s[d+n*INDDIM] = INDARG[d+INDARG_map[n]*INDDIM];';
            elseif(indaccs(m)==OP_INC)
              line = '      INDARG_s[d+n*INDDIM] = ZERO_INDTYP;';
            end
            file = strvcat(file,rep(line,m),' ');
          end
        end
      end

      file = strvcat(file,'  __syncthreads();',' ',...
                          '  // process set elements',' ');

      if (ind_inc)
        if (target==OP_CUDA)
	  file = strvcat(file,...
              '  for (int n=threadIdx.x; n<nelems2; n+=blockDim.x) {');
        elseif (target==OP_x86)
          file = strvcat(file,...
               '  for (int n=0; n<nelems2; n++) {');
        end
        file = strvcat(file,...
               '    int col2 = -1;                             ',' ',...
               '    if (n<nelem) {                             ',' ',...
               '      // initialise local variables            ',' ');

        for m = 1:nargs
          if (maps(m)==OP_MAP & accs(m)==OP_INC)
            line = '      for (int d=0; d<DIM; d++)';
            file = strvcat(file,rep(line,m));
            line = '        ARG_l[d] = ZERO_TYP;';
            file = strvcat(file,rep(line,m));
          end
        end

      else
        if (target==OP_CUDA)
          file = strvcat(file,...
                 '  for (int n=threadIdx.x; n<nelem; n+=blockDim.x) {');
        elseif (target==OP_x86)
          file = strvcat(file,...
                 '  for (int n=0; n<nelem; n++) {');
        end
      end

%
% simple alternative when no indirection
%
    else

     if (target==OP_CUDA)
      use_shared = 0;
      for m = 1:nargs
       if(maps(m)~=OP_GBL && dims{m}~='1')
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
       if(maps(m)~=OP_GBL && accs(m)~=OP_WRITE && dims{m}~='1')
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

     elseif (target==OP_x86)
      file = strvcat(file,' ','  // process set elements',' ', ...
                          '  for (int n=start; n<finish; n++) {');
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
    
    file = strvcat(file,' ',...
                   [ prefix '// user-supplied kernel call'],' ');

    for m = 1:nargs
      line = [prefix fn_name '( '];
      if (m~=1)
        line = blanks(length(line));
      end
      if (maps(m)==OP_GBL)
        if (accs(m)==OP_READ || target==OP_x86 )
          line = [ line 'ARG,' ];
        else
          line = [ line 'ARG_l,' ];
        end
      elseif (maps(m)==OP_MAP & accs(m)==OP_INC)
        line = [ line 'ARG_l,' ];
      elseif (maps(m)==OP_MAP)
        line = [ line ...
        sprintf('ind_arg%d_s+ARG_maps[n+offset_b]*DIM,',inds(m)-1) ];
      elseif (maps(m)==OP_ID)
        if (ninds>0)
          line = [ line 'ARG+(n+offset_b)*DIM,' ];
        else
          if (target==OP_CUDA)
            if (dims{m}=='1')
              line = [ line 'ARG+n,' ];
            else
              line = [ line 'ARG_l,' ];
            end

          elseif(target==OP_x86)
            line = [ line 'ARG+n*DIM,' ];
          end
        end

      else
        error('internal error 1')
      end
      if (m==nargs)
        line = [ line(1:end-1) ' );' ];
      end

      file = strvcat(file,rep(line,m));
    end

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
          if (maps(m)==OP_MAP & accs(m)==OP_INC)
            line = sprintf('    int ARG_map = ARG_maps[n+offset_b];');
            file = strvcat(file,rep(line,m));
          end
        end

        file = strvcat(file,...
              ' ','    for (int col=0; col<ncolor; col++) {',...
                  '      if (col2==col) {                  ');
        for m = 1:nargs
          if (maps(m)==OP_MAP & accs(m)==OP_INC)
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
        if(indaccs(m)==OP_WRITE | indaccs(m)==OP_RW | indaccs(m)==OP_INC)
          if (target==OP_CUDA)
            line = '  for (int n=threadIdx.x; n<INDARG_size*INDDIM; n+=blockDim.x)';
            file = strvcat(file,rep(line,m));
            if(indaccs(m)==OP_WRITE | indaccs(m)==OP_RW)
              line = '    INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] = INDARG_s[n];';
              file = strvcat(file,rep(line,m),' ');
            elseif(indaccs(m)==OP_INC)
              line = '    INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] += INDARG_s[n];';
              file = strvcat(file,rep(line,m),' ');
            end
          elseif (target==OP_x86)
            line = '  for (int n=0; n<INDARG_size; n++)';
            file = strvcat(file,rep(line,m));
            line = '    for (int d=0; d<INDDIM; d++)';
            file = strvcat(file,rep(line,m));
            if(indaccs(m)==OP_WRITE | indaccs(m)==OP_RW)
              line = '      INDARG[d+INDARG_map[n]*INDDIM] = INDARG_s[d+n*INDDIM];';
              file = strvcat(file,rep(line,m),' ');
            elseif(indaccs(m)==OP_INC)
              line = '      INDARG[d+INDARG_map[n]*INDDIM] += INDARG_s[d+n*INDDIM];';
              file = strvcat(file,rep(line,m),' ');
            end
          end
        end
      end
%
% ... and direct kernels
%
    else

      if (target==OP_CUDA)
        if (use_shared)
          file = strvcat(file,' ',...
             '    // copy back into shared memory, then to device',' ');
        end

        for m = 1:nargs
          if(maps(m)~=OP_GBL && accs(m)~=OP_READ && dims{m}~='1')
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
      end

      file = strvcat(file,'  }');
    end

%
% global reduction
%
    if (target==OP_CUDA & reduct)
      file = strvcat(file,' ','  // global reductions',' ');
      for m = 1:nargs
        if (maps(m)==OP_GBL & accs(m)~=OP_READ)
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
          ['void op_par_loop_' fn_name '(char const *name, op_set set,']);

    for m = 1:nargs
      line = rep('  op_arg ARG', m);

      if (m<nargs)
        file = strvcat(file,[line ',']);
      else
        file = strvcat(file,[line ' ){'],' ');
      end
    end

    for m = 1:nargs
      if (maps(m)==OP_GBL)
        line = '  TYP *ARGh = (TYP *)ARG.data;';
        file = strvcat(file,rep(line,m));
      end
    end

%
%   indirect bits
%
    if (ninds>0)
      file = strvcat(file,' ',['  int    nargs   = ' num2str(nargs) ';']);

      line = sprintf('  op_arg args[%d] = {',nargs);
      for m = 1:nargs
        line = [line 'arg' num2str(m-1) ','];
      end
      file = strvcat(file,[line(1:end-1) '};']);

      file = strvcat(file,' ',['  int    ninds   = ' num2str(ninds) ';']);

      line = sprintf('  int    inds[%d] = {',nargs);
      for m = 1:nargs
        line = strcat(line,num2str(inds(m)-1),',');
      end
      file = strvcat(file,[line(1:end-1) '};']);

      file = strvcat(file,' ',...
      '  if (OP_diags>2) {              ',...
     ['    printf(" kernel routine with indirection: ' fn_name ' \n");'],...
      '  }                              ',' ',...
      '  // get plan                    ',' ',...
     ['  #ifdef OP_PART_SIZE_'            num2str(nker)    ],...
     ['    int part_size = OP_PART_SIZE_' num2str(nker) ';'],...
      '  #else                          ',...
      '    int part_size = OP_part_size;',...
      '  #endif                         ',' ',...
      '  op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);');

%
% direct bit
%
    else
      file = strvcat(file,' ',...
       '  if (OP_diags>2) {              ',...
      ['    printf(" kernel routine w/o indirection:  ' fn_name ' \n");'],...
       '  }                              ');
    end

%
% start timing
%
    file = strvcat(file,' ','  // initialise timers                    ',...
                        ' ','  double cpu_t1, cpu_t2, wall_t1, wall_t2;',...
                            '  op_timers(&cpu_t1, &wall_t1);           ');

%
% set number of threads in x86 execution and create arrays for reduction
%
    if (target==OP_x86)
      file = strvcat(file,' ','  // set number of threads',' ',...
                   '#ifdef _OPENMP                          ',...
                   '  int nthreads = omp_get_max_threads( );',...
                   '#else                                   ',...
                   '  int nthreads = 1;                     ',...
                   '#endif                                  ');

      if (reduct)
       file = strvcat(file,' ',...
             '  // allocate and initialise arrays for global reduction');
       for m = 1:nargs
        if (maps(m)==OP_GBL & accs(m)~=OP_READ)
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
    end
%
% transfer constants
%
    if (target==OP_CUDA)

    if (length(find(maps(1:nargs)==OP_GBL & accs(1:nargs)==OP_READ))>0)
      file = strvcat(file,'  ',...
       '  // transfer constants to GPU',' ',...
       '  int consts_bytes = 0;');
      for m=1:nargs
        if(maps(m)==OP_GBL & accs(m)==OP_READ);
          line = '  consts_bytes += ROUND_UP(DIM*sizeof(TYP));';
          file = strvcat(file,rep(line,m));
        end
      end

      file = strvcat(file,'  ',...
       '  reallocConstArrays(consts_bytes);',' ',...
       '  consts_bytes = 0;');

      for m=1:nargs
        if(maps(m)==OP_GBL & accs(m)==OP_READ);
          line = '  ARG.data   = OP_consts_h + consts_bytes;';
          file = strvcat(file,rep(line,m));
          line = '  ARG.data_d = OP_consts_d + consts_bytes;';
          file = strvcat(file,rep(line,m));
          line = ...
   '  for (int d=0; d<DIM; d++) ((TYP *)ARG.data)[d] = ARGh[d];';
          file = strvcat(file,rep(line,m));
          line = '  consts_bytes += ROUND_UP(DIM*sizeof(TYP));';
          file = strvcat(file,rep(line,m));
        end
      end

      file = strvcat(file,'  ','  mvConstArraysToDevice(consts_bytes);');
    end

    end

%
% transfer global reduction initial data
%
    if (target==OP_CUDA)

    if (ninds==0)
      file = strvcat(file,' ',...
        '  // set CUDA execution parameters  ',' ',...
       ['  #ifdef OP_BLOCK_SIZE_'          num2str(nker)    ],...
       ['    int nthread = OP_BLOCK_SIZE_' num2str(nker) ';'],...
        '  #else                             ',...
        '    // int nthread = OP_block_size; ',...
        '    int nthread = 128;              ',...
        '  #endif                            ',' ',...
        '  int nblocks = 200;                ');
    end

    if (reduct)
      file = strvcat(file,'  ',...
           '  // transfer global reduction data to GPU',' ');

      if (ninds>0)
        file = strvcat(file,...
           '  int maxblocks = 0;',...
           '  for (int col=0; col < Plan->ncolors; col++)',...
           '    maxblocks = MAX(maxblocks,Plan->ncolblk[col]);');
      else
        file = strvcat(file,'  int maxblocks = nblocks;');
      end

      file = strvcat(file,' ','  int reduct_bytes = 0;',...
                              '  int reduct_size  = 0;');

      for m=1:nargs
        if(maps(m)==OP_GBL & accs(m)~=OP_READ);
          line = '  reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));';
          file = strvcat(file,rep(line,m));
          line = '  reduct_size   = MAX(reduct_size,sizeof(TYP));';
          file = strvcat(file,rep(line,m));
        end
      end

      file = strvcat(file,'  ',...
       '  reallocReductArrays(reduct_bytes);',' ',...
       '  reduct_bytes = 0;');

      for m=1:nargs
        if(maps(m)==OP_GBL & accs(m)~=OP_READ);
          line = '  ARG.data   = OP_reduct_h + reduct_bytes;';
          file = strvcat(file,rep(line,m));
          line = '  ARG.data_d = OP_reduct_d + reduct_bytes;';
          file = strvcat(file,rep(line,m));
          file = strvcat(file,'  for (int b=0; b<maxblocks; b++)');
          line = '    for (int d=0; d<DIM; d++)';
          file = strvcat(file,rep(line,m));
          if (accs(m)==OP_INC)
            line = '      ((TYP *)ARG.data)[d+b*DIM] = ZERO_TYP;';
          else
            line = '      ((TYP *)ARG.data)[d+b*DIM] = ARGh[d];';
          end
          file = strvcat(file,rep(line,m));
          line = '  reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));';
          file = strvcat(file,rep(line,m));
        end
      end

      file = strvcat(file,'  ','  mvReductArraysToDevice(reduct_bytes);');
    end

    end

%
% kernel call for indirect version
%
    if (ninds>0)

      if (target==OP_CUDA)
       file = strvcat(file,' ',...
        '  // execute plan                ',' ',...
        '  int block_offset = 0;          ',' ',...
        '  for (int col=0; col < Plan->ncolors; col++) { ',' ',...
       ['  #ifdef OP_BLOCK_SIZE_'          num2str(nker)    ],...
       ['    int nthread = OP_BLOCK_SIZE_' num2str(nker) ';'],...
        '  #else                          ',...
        '    int nthread = OP_block_size; ',...
        '  #endif                         ',' ',...
        '    int nblocks = Plan->ncolblk[col];         ');

       if (reduct)
 	file = strvcat(file,...
        '    int nshared = MAX(Plan->nshared,reduct_size*nthread);');
       else
 	file = strvcat(file,'    int nshared = Plan->nshared;');
       end

       file = strvcat(file,...
       ['    op_cuda_' fn_name '<<<nblocks,nthread,nshared>>>(']);

       for m = 1:ninds
        line = sprintf('       (TYP *)ARG.data_d, Plan->ind_maps[%d],',m-1);
        file = strvcat(file,rep(line,invinds(m)));
       end

       for m = 1:nargs
         if (inds(m)==0)
           line = '       (TYP *)ARG.data_d,';
         else
           line = sprintf('       Plan->loc_maps[%d],',m-1);
         end
         file = strvcat(file,rep(line,m));
       end

       file = strvcat(file, ... 
       '       Plan->ind_sizes,                                     ',...
       '       Plan->ind_offs,                                      ',...
       '       block_offset,                                        ',...
       '       Plan->blkmap,                                        ',...
       '       Plan->offset,                                        ',...
       '       Plan->nelems,                                        ',...
       '       Plan->nthrcol,                                       ',...
       '       Plan->thrcol);                                       ',...
   ' ','    cutilSafeCall(cudaThreadSynchronize());                 ',...
      ['    cutilCheckMsg("op_cuda_' fn_name ' execution failed\n");'],...
   ' ','    block_offset += nblocks;                                ',...
       '  }                                                         ');

      elseif (target==OP_x86)
       file = strvcat(file,' ',...
        '  // execute plan                             ',' ',...
        '  int block_offset = 0;                       ',' ',...
        '  for (int col=0; col < Plan->ncolors; col++) {   ',...
        '    int nblocks = Plan->ncolblk[col];         ',' ',...
        '#pragma omp parallel for',...
        '    for (int blockIdx=0; blockIdx<nblocks; blockIdx++)');

       file = strvcat(file,['     op_x86_' fn_name '( blockIdx,']);

       for m = 1:ninds
        line = sprintf('       (TYP *)ARG.data, Plan->ind_maps[%d],',m-1);
        file = strvcat(file,rep(line,invinds(m)));
       end

       for m = 1:nargs
         if (inds(m)==0)
           line = '       (TYP *)ARG.data,';
         else
           line = sprintf('       Plan->loc_maps[%d],',m-1);
         end
         file = strvcat(file,rep(line,m));
       end

       file = strvcat(file, ... 
        '       Plan->ind_sizes,                              ',...
        '       Plan->ind_offs,                               ',...
        '       block_offset,                                 ',...
        '       Plan->blkmap,                                 ',...
        '       Plan->offset,                                 ',...
        '       Plan->nelems,                                 ',...
        '       Plan->nthrcol,                                ',...
        '       Plan->thrcol);                                ',' ',...
        '    block_offset += nblocks;                         ',...
        '  }                                                  ');
      end
%
% kernel call for direct version
%
    else
        if (target==OP_CUDA)
        file = strvcat(file,...
          ' ','  // work out shared memory requirements per element',...
          ' ','  int nshared = 0;');

        for m = 1:nargs
          if(maps(m)~=OP_GBL && dims{m}~='1');
            line = '  nshared = MAX(nshared,sizeof(TYP)*DIM);';
            file = strvcat(file,rep(line,m));
          end
        end

        file = strvcat(file,...
          ' ','  // execute plan                    ',' ',...
              '  int offset_s = nshared*OP_WARPSIZE;',' ');

        if (reduct)
          file = strvcat(file,...
           '  nshared = MAX(nshared*nthread,reduct_size*nthread);',' ');
        else
          file = strvcat(file,'  nshared = nshared*nthread;',' ');
        end
        line = ['  op_cuda_' fn_name '<<<nblocks,nthread,nshared>>>( '];

        for m = 1:nargs
          file = strvcat(file,rep([line '(TYP *) ARG.data_d,'],m));
          line = blanks(length(line));
        end

        file = strvcat(file,[ line 'offset_s,'  ],... 
                            [ line 'set->size );'],' ',... 
        '  cutilSafeCall(cudaThreadSynchronize());                ', ...
       ['  cutilCheckMsg("op_cuda_', fn_name ' execution failed\n");']);

      elseif (target==OP_x86)
        file = strvcat(file,...
           ' ','  // execute plan                            ',...
           ' ','#pragma omp parallel for                     ',...
               '  for (int thr=0; thr<nthreads; thr++) {     ',...
               '    int start  = (set->size* thr   )/nthreads;',...
               '    int finish = (set->size*(thr+1))/nthreads;');
        line = ['    op_x86_' fn_name '( '];

        for m = 1:nargs
          if(maps(m)==OP_GBL & accs(m)~=OP_READ);
            file = strvcat(file,rep([line 'ARG_l + thr*64,'],m));
          else
            file = strvcat(file,rep([line '(TYP *) ARG.data,'],m));
          end
          line = blanks(length(line));
        end

        file = strvcat(file,[ line 'start, finish );'],'  }');
      end
    end

%
% transfer global reduction initial data
%
    if (target==OP_CUDA && reduct)
      file = strvcat(file,...
           ' ','  // transfer global reduction data back to CPU',...
           ' ','  mvReductArraysToHost(reduct_bytes);',' ');
      for m=1:nargs
        if(maps(m)==OP_GBL & accs(m)~=OP_READ);
         file = strvcat(file,'  for (int b=0; b<maxblocks; b++)');
         line = '    for (int d=0; d<DIM; d++)';
         file = strvcat(file,rep(line,m));
         if (accs(m)==OP_INC)
          line = '      ARGh[d] = ARGh[d] + ((TYP *)ARG.data)[d+b*DIM];';
         elseif (accs(m)==OP_MIN)
          line = '      ARGh[d] = MIN(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);';
         elseif (accs(m)==OP_MAX)
          line = '      ARGh[d] = MAX(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);';
         end
         file = strvcat(file,rep(line,m));
        end
      end
    end

%
% combine reduction data from multiple OpenMP threads
%
    if (target==OP_x86 && reduct)
     file = strvcat(file,' ','  // combine reduction data');
     for m=1:nargs
      if(maps(m)==OP_GBL & accs(m)~=OP_READ);
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
      end
     end
    end

%
% update kernel record
%

  file = strvcat(file,' ','  // update kernel record',' ',...
     '  op_timers(&cpu_t2, &wall_t2);                               ',...
    ['  op_timing_realloc(' num2str(nker) ');                       '],...
    ['  OP_kernels[' num2str(nker) '].name      = name;             '],...
    ['  OP_kernels[' num2str(nker) '].count    += 1;                '],...
    ['  OP_kernels[' num2str(nker) '].time     += wall_t2 - wall_t1;']);
  if (ninds>0)
   file = strvcat(file,...
    ['  OP_kernels[' num2str(nker) '].transfer  += Plan->transfer; '],...
    ['  OP_kernels[' num2str(nker) '].transfer2 += Plan->transfer2;']);
  else

   line = ...
    ['  OP_kernels[' num2str(nker) '].transfer += (float)set->size *'];

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

    if (target==OP_CUDA)
      fid = fopen(strcat(fn_name,'_kernel.cu'),'wt');
    elseif (target==OP_x86) 
      fid = fopen(strcat(fn_name,'_kernel.cpp'),'wt');
    end

    fprintf(fid,'// \n// auto-generated by op2.m on %s \n//\n\n',date);
    for n=1:size(file,1)
      line = file(n,:);
      if (target==OP_x86)
        line = regexprep(line,'__shared__ ','');
      end
      fprintf(fid,'%s\n',line);
    end
    fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  append kernel bits for master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (nkernels==1 & narg==1)
      ker_file3 = ' ';
    end

    if (target==OP_CUDA)
     ker_file3 = strvcat(ker_file3,['#include "' fn_name '_kernel.cu"']);
    elseif (target==OP_x86)
     ker_file3 = strvcat(ker_file3,['#include "' fn_name '_kernel.cpp"']);
    end

  end  % end of main kernel call loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  process global constants for master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if(narg==1)
    ker_file1 = '';
    ker_file2 = '';
  end

  for const_index = 1:n_consts
    name = consts{const_index}.name;
    if (name(1)=='&')
      name = name(2:end);
      consts{const_index}.name = name;
    end

    type = consts{const_index}.type(2:end-1);

    [dim,ok] = str2num(consts{const_index}.dim);
    if (ok==0)
      dim = -999;
    end

    repeat = 0;
    for c = 1:nconsts
      if (strcmp(name,global_consts{c}{1}))
        repeat = 1;
        if (~strcmp(type,global_consts{c}{2}))
          error(sprintf('type mismatch in repeated op_decl_const'));
        end
        if (dim ~= global_consts{c}{3})
          error(sprintf('size mismatch in repeated op_decl_const'));
        end
      end
    end

    if (repeat)
     disp(sprintf('\n  repeated global constant (%s)',name));

    else
     nconsts = nconsts +1;
     global_consts{nconsts}{1} = name;
     global_consts{nconsts}{2} = type;
     global_consts{nconsts}{3} = dim;

     if (target==OP_CUDA)
  
      if (ok & dim==1)
       ker_file1 = strvcat(ker_file1, ...
         [ '__constant__ ' type ' ' name ';' ]);
      elseif (ok)
       ker_file1 = strvcat(ker_file1, ...
     [ '__constant__ ' type ' ' name '[' num2str(dim) '];' ]);
      else
       ker_file1 = strvcat(ker_file1, ...
         [ '__constant__ ' type ' ' name '[MAX_CONST_SIZE];' ]);
       ker_file2 = strvcat(ker_file2, ...
     ['  if(~strcmp(name,"' name '") && size>MAX_CONST_SIZE) {'],...
     ['    printf("error: MAX_CONST_SIZE not big enough\n"); exit(1);'],...
      '  }');

      end

     elseif (target==OP_x86)
      if (ok & dim==1)
       ker_file1 = strvcat(ker_file1, ...
         [ 'extern ' type ' ' name ';' ]);
      else
       ker_file1 = strvcat(ker_file1, ...
         [ 'extern ' type ' ' name '[' num2str(dim) '];' ]);
      end
     end

     disp(sprintf('\n  global constant (%s) of size %s',...
                  name,consts{const_index}.dim));
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  output new source file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if (target==1)
    fid = fopen(strcat(filename,'_op.cpp'),'wt');
    fprintf(fid,'// \n// auto-generated by op2.m on %s \n//\n\n',date);

    loc_old = 1;

    loc_header = strfind(src_file,'"op_seq.h"');

    locs = sort([ loc_header loc_loops loc_consts]);

    for loc = locs
      fprintf(fid,'%s',src_file(loc_old:loc-1));
      loc_old = loc-1;

      if (~isempty(find(loc==loc_header)))
        fprintf(fid,'"op_lib.h"\n\n');
        fprintf(fid,'//\n// op_par_loop declarations\n//\n');

        for k=1:nkernels
	  fprintf(fid,'\nvoid op_par_loop_%s(char const *, op_set,\n',...
                  loop_args{k}.name1);
          for n = 1:loop_args{k}.nargs-1
            fprintf(fid,'  op_arg,\n');
          end
          fprintf(fid,'  op_arg );\n');
        end

        loc_old = loc+14;
      end

      if (~isempty(find(loc==loc_loops)))
        name = loop_args{find(loc==loc_loops)}.name1;
        while (src_file(loc)~=',' || ~active(src_file(loc_old:loc)))
          loc = loc+1;
        end
        fprintf(fid,'_%s(',name);
        loc_old = loc+1;
      end

      if (~isempty(find(loc==loc_consts)))
        name = consts{find(loc==loc_consts)}.name;
        while (src_file(loc)~='(' || ~active(src_file(loc_old:loc)))
          loc = loc+1;
        end
        fprintf(fid,'2("%s",',name);
        loc_old = loc+1;
      end

    end

    fprintf(fid,'%s',src_file(loc_old:end));

    fclose(fid);
  end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  output one master kernel file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (target==OP_CUDA)
  file = strvcat(...
    '// header                 ',' ',...
    '#include "op_lib.cu"      ',' ',...
    '// global constants       ',' ',...
    '#ifndef MAX_CONST_SIZE    ',...
    '#define MAX_CONST_SIZE 128',...
    '#endif                    ',' ',...
    ker_file1,...
    ' ',...
    'void op_decl_const_char(int dim, char const *type,',...
    '            int size, char *dat, char const *name){',...
    ker_file2,...
    '  cutilSafeCall(cudaMemcpyToSymbol(name, dat, dim*size));',....
    '} ',' ',...
    '// user kernel files',...
    ker_file3);

  fid = fopen([ varargin{1} '_kernels.cu'],'wt');

elseif (target==OP_x86) 
  file = strvcat(...
    '// header                 ',' ',...
    '#include "op_lib.h"       ',' ',...
    'void __syncthreads(){}    ',' ',...
    '// global constants       ',' ',...
    ker_file1,' ',...
    '// user kernel files',...
    ker_file3);

  fid = fopen([ varargin{1} '_kernels.cpp'],'wt');
end

fprintf(fid,'// \n// auto-generated by op2.m on %s \n//\n\n',date);

for n=1:size(file,1)
  fprintf(fid,'%s\n',file(n,:));
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
global OP_typs_labels OP_typs_CPP

line = regexprep(line,'INDDIM',inddims(m));
line = regexprep(line,'INDARG',sprintf('ind_arg%d',m-1));
line = regexprep(line,'INDTYP',indtyps(m));

line = regexprep(line,'DIM',dims(m));
line = regexprep(line,'ARG',sprintf('arg%d',m-1));
line = regexprep(line,'TYP',typs(m));
line = regexprep(line,'IDX',num2str(idxs(m)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% parsing for op_decl_const calls
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function consts = op_decl_const_parse(file)

consts = [];

locs = strfind(file,'op_decl_const');

for n = 1:length(locs)
  loc = locs(n);
  if active(file(1:loc))
    loc = loc + length('op_decl_const');
    args_str = active_compress(file(loc:end));

    try
      args = args_parse(args_str);
      consts{n}.loc  = loc;
      consts{n}.dim  = args{1};
      consts{n}.type = args{2};
      consts{n}.name = args{3};
    catch
      error(sprintf('error parsing op_decl_const'));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% parsing for op_par_loop calls
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function loop_args = op_par_loop_parse(file)

locs = strfind(file,'op_par_loop');

for n = 1:length(locs)
  loc = locs(n);
  if active(file(1:loc))
    loc = loc + length('op_par_loop');
    args_str = active_compress(file(loc:end));

    try
      args = args_parse(args_str);
      loop_args{n}.loc   = loc;
      loop_args{n}.name1 = args{1};
      loop_args{n}.name2 = args{2};
      loop_args{n}.set   = args{3};
      loop_args{n}.nargs = length(args)-3;
      for m = 1:length(args)-3
        if     strcmp(args{m+3}(1:10),'op_arg_dat')
          loop_args{n}.type{m} = 'op_arg_dat';
        elseif strcmp(args{m+3}(1:10),'op_arg_gbl')
          loop_args{n}.type{m} = 'op_arg_gbl';
        end
        loop_args{n}.args{m} = args{m+3}(11:end);
      end
    catch
      error(sprintf('error parsing op_par_loop'));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% parse a single set of arguments
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function args = args_parse(string)

if ~strcmp(string(1),'(')
  error
end

loc1 = 2;
loc2 = 1;
depth = 0;
nargs = 0;

while 1
  switch string(loc2)
    case '('
      depth = depth + 1;
    case ')'
      depth = depth - 1;
      if (depth==0)
        nargs = nargs + 1;
        args{nargs} = string(loc1:loc2-1);
        return
      end
    case ','
      if (depth==1)
        nargs = nargs + 1;
        args{nargs} = string(loc1:loc2-1);
        loc1 = loc2 + 1;
      end
  end
  loc2 = loc2+1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% check if the last character in a C++/C99 file is active,
% i.e. not commented out
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ans = active(file)

active_list = active_find(file);
ans = (active_list(end) == length(file));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% compress a C++/C99 file, removing the comments
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ans = active_compress(file)

active_list = active_find(file);
ans = [];
for m = 1:size(active_list,2)
  ans = [ ans file(active_list(1,m):active_list(2,m)) ];
end

ans = regexprep(ans,'\s','');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% routine to determine the active parts of a C++/C99 file,
% i.e. the parts which are not commented out
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function active_list = active_find(file)

len = length(file);

loc1 = [ strfind(file,'/*') len+1 ];
loc2 = [ strfind(file,'//') len+1 ];
loc3 = [ strfind(file,'*/') len+1 ];
loc4 = [ strfind(file,sprintf('\n')) len+1 ];

start = 1;

active_list = [];

while (start<=len)
  next = min(loc1(1),loc2(1));

% ignore cases embedded within a text string
  if (next<=len)
    while (mod( length(strfind(file(start:next),'"')), 2)==1)
      loc1 = loc1(find(loc1>next));
      loc2 = loc2(find(loc2>next));
      next = min(loc1(1),loc2(1));
    end
  end

  if (next>start)
    active_list = [ active_list [start; next-1] ];
  end

  if (next==loc1(1))
    %  /* terminated by */
    start = loc3(min(find(loc3>next))) + 2;
  else
    %  // terminated by newline
    start = loc4(min(find(loc4>next))) + 1;
  end

  loc1 = loc1(find(loc1>=start));
  loc2 = loc2(find(loc2>=start));
end


