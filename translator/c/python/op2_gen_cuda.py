##########################################################################
#
# CUDA code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.cu for each kernel,
# plus a master kernel file
#
##########################################################################

import datetime

def op2_gen_cuda(master, date, consts, kernels):

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;
  
  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]
  
  any_soa = 0
  for nk in range (0,len(kernels)):
    any_soa = any_soa or sum(kernels[nk]['soaflags'])    

##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):
      
    name = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dims  = kernels[nk]['dims']
    maps  = kernels[nk]['maps']
    var  = kernels[nk]['var']
    typs  = kernels[nk]['typs']
    accs  = kernels[nk]['accs']
    idxs  = kernels[nk]['idxs']
    inds  = kernels[nk]['inds']
    soaflags = kernels[nk]['soaflags']

    ninds   = kernels[nk]['ninds']
    inddims = kernels[nk]['inddims']
    indaccs = kernels[nk]['indaccs']
    indtyps = kernels[nk]['indtyps']
    invinds = kernels[nk]['invinds']
    
    vec =  [m for m in range(0,nargs) if int(idxs[m])<0 and maps[m] == OP_MAP]
    
    if len(vec) > 0:
      unique_args = [1];
      vec_counter = 1;
      vectorised = []
      new_dims = []
      new_maps = []
      new_vars = []
      new_typs = []
      new_accs = []
      new_idxs = []
      new_inds = []
      new_soaflags = []
      for m in range(0,nargs):
      	  if int(idxs[m])<0 and maps[m] == OP_MAP:
      	    if m > 0:
      	      unique_args = unique_args + [len(new_dims)+1]
      	    temp = [0]*(-1*int(idxs[m]))
      	    for i in range(0,-1*int(idxs[m])): 
      	      temp[i] = var[m]
      	    new_vars = new_vars+temp
      	    for i in range(0,-1*int(idxs[m])): 
      	      temp[i] = typs[m]
      	    new_typs = new_typs+temp
      	    for i in range(0,-1*int(idxs[m])): 
      	      temp[i] = dims[m]
      	    new_dims = new_dims+temp
      	    new_maps = new_maps+[maps[m]]*int(-1*int(idxs[m]))
      	    new_soaflags = new_soaflags+[0]*int(-1*int(idxs[m]))
      	    new_accs = new_accs+[accs[m]]*int(-1*int(idxs[m]))
            for i in range(0,-1*int(idxs[m])):
            	new_idxs = new_idxs+[i]
            new_inds = new_inds+[inds[m]]*int(-1*int(idxs[m]))  
            vectorised = vectorised + [vec_counter]*int(-1*int(idxs[m]))
            vec_counter = vec_counter + 1;
          else:
            if m > 0:
              unique_args = unique_args + [len(new_dims)+1]
            new_dims = new_dims+[dims[m]]
            new_maps = new_maps+[maps[m]]
            new_accs = new_accs+[int(accs[m])]
            new_soaflags = new_soaflags+[soaflags[m]]
            new_idxs = new_idxs+[int(idxs[m])]
            new_inds = new_inds+[inds[m]]
            new_vars = new_vars+[var[m]]
            new_typs = new_typs+[typs[m]]
            vectorised = vectorised+[0]
      dims = new_dims
      maps = new_maps
      accs = new_accs
      idxs = new_idxs
      inds = new_inds
      var = new_vars
      typs = new_typs
      soaflags = new_soaflags;
      nargs = len(vectorised);
      
      for i in range(1,ninds+1):
      	for index in range(0,len(inds)+1):
      	  if inds[index] == i:
      	    invinds[i-1] = index
      	    break      
    else:
    	vectorised = [0]*nargs
    	unique_args = range(1,nargs+1)

    cumulative_indirect_index = [-1]*nargs;
    j = 0;
    for i in range (0,nargs):
      if maps[i] == OP_MAP:
        cumulative_indirect_index[i] = j
        j = j + 1        
#
# set two logicals
#
    j = 0
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
    ind_inc = j > 0
    
    j = 0
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ:
        j = i
    reduct = j > 0

##########################################################################
#  start with CUDA kernel function
##########################################################################

    file_text = '//user function\n'\
    '__device__\n'\
    '#include "'+name+'.h"\n\n'\
    '// CUDA kernel function\n\n'\
    '__global__ void op_cuda_'+name+'(\n'

    for m in range(0,ninds):
      file_text = file_text+'  '+str(indtyps[m])+' *ind_arg'+str(m)+',\n'

    if ninds>0:
      file_text = file_text+'  int   *ind_map,\n'
      file_text = file_text+'  short *arg_map,\n'
  
    for m in range (0,nargs):
      if maps[m]==OP_GBL and accs[m] == OP_READ:
        file_text = file_text+'  const '+typs[m]+' *arg'+str(m)+',\n'  # declared const for performance
      elif maps[m]==OP_ID and ninds>0:
        file_text = file_text+'  '+typs[m]+' *arg'+str(m)+',\n'
      elif maps[m]==OP_GBL or maps[m]==OP_ID:
        file_text = file_text+'  '+typs[m]+' *arg'+str(m)+',\n'
    
    if ninds>0:
      file_text = file_text+'  int   *ind_arg_sizes,\n'\
      '  int   *ind_arg_offs,\n'\
      '  int    block_offset,\n'\
      '  int   *blkmap,      \n'\
      '  int   *offset,      \n'\
      '  int   *nelems,      \n'\
      '  int   *ncolors,     \n'\
      '  int   *colors,      \n'\
      '  int   nblocks,     \n'\
      '  int   set_size) {   \n\n'
    else:
      file_text = file_text+'  int   offset_s,    \n'\
                            '  int   set_size ) {\n\n'
    
    for m in range(0,nargs):
      if maps[m]==OP_GBL and accs[m]<>OP_READ:
        file_text = file_text+'  '+typs[m]+' arg'+str(m)+'_l['+dims[m]+'];\n'
        if accs[m] == OP_INC:
          file_text = file_text+\
          '  for (int d=0; d<'+dims[m]+'; d++) arg'+str(m)+'_l[d]=ZERO_'+typs[m]+';\n'
        else:
          file_text = file_text+\
          '  for (int d=0; d<'+dims[m]+'; d++) arg'+str(m)+'_l[d]=arg'+str(m)+'[d+blockIdx.x*'+dims[m]+'];\n'
      elif maps[m]==OP_MAP and accs[m]==OP_INC:
        file_text = file_text+'  '+typs[m]+'  arg'+str(m)+'_l['+str(dims[m])+'];\n'
      elif (ninds==0 and maps[m]==OP_ID and dims[m]<>'1') and not(soaflags[m]):
        file_text = file_text+'  '+typs[m]+'  arg'+str(m)+'_l['+str(dims[m])+'];\n'
    
    file_text = file_text+'\n' 
    
    for m in range (1,ninds+1):
      v = [int(inds[i]==m) for i in range(len(inds))]
      v_i = [vectorised[i] for i in range(len(inds)) if inds[i] == m] 
      if sum(v)>1 and sum(v_i)>0: #check this sum(v_i) 
        if indaccs[m-1] == OP_INC:
          ind = int(max([idxs[i] for i in range(len(inds)) if inds[i]==m])) + 1
          file_text = file_text+'  '+indtyps[m-1]+' *arg'+str(m-1)+'_vec['+str(ind)+'] = {\n'
          for n in range(0,nargs):
            if inds[n] == m:
              file_text = file_text+'    arg'+str(n)+'_l,\n'
              
          file_text = file_text[0:-2]+'\n  };\n'
        else:
          ind = int(max([idxs[i] for i in range(len(inds)) if inds[i]==m])) + 1
          file_text = file_text+'  '+indtyps[m-1]+' *arg'+str(m-1)+'_vec['+str(ind)+'];\n'
# 
# lengthy code for general case with indirection
#
    if ninds>0:
      file_text = file_text+' \n'
      for m in range (0,ninds):
        file_text = file_text+'  __shared__  int  *ind_arg'+str(m)+'_map, ind_arg'+str(m)+'_size;\n'
      for m in range (0,ninds):
        file_text = file_text+'  __shared__  '+indtyps[m]+' *ind_arg'+str(m)+'_s;\n'
      
      if ind_inc:
        file_text = file_text+'  __shared__ int    nelems2, ncolor;\n'
      
      file_text = file_text+\
        '  __shared__ int    nelem, offset_b;\n\n'\
        '  extern __shared__ char shared[];\n\n'\
        '  if (blockIdx.x+blockIdx.y*gridDim.x >= nblocks) return;\n'\
        '  if (threadIdx.x==0) {\n\n'\
        '    // get sizes and shift pointers and direct-mapped data\n\n'\
        '    int blockId = blkmap[blockIdx.x + blockIdx.y*gridDim.x  + block_offset];\n\n'\
        '    nelem    = nelems[blockId];\n'\
        '    offset_b = offset[blockId];\n\n'
      
      if ind_inc:
        file_text = file_text+\
        '    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);\n'\
        '    ncolor   = ncolors[blockId];\n\n'
      
      for m in range (0,ninds):
        file_text = file_text+\
        '    ind_arg'+str(m)+'_size = ind_arg_sizes['+str(m)+'+blockId*'+ str(ninds)+'];\n'
      
      file_text = file_text+'\n'
       
      for m in range (1,ninds+1):
        c = [i for i in range(len(inds)) if inds[i]==m]
        file_text = file_text+\
        '    ind_arg'+str(m-1)+'_map = &ind_map['+str(cumulative_indirect_index[c[0]])+\
        '*set_size] + ind_arg_offs['+str(m-1)+'+blockId*'+str(ninds)+'];\n'
       
      file_text = file_text+'\n'\
      '    // set shared memory pointers\n'\
      '    int nbytes = 0;\n'
       
      for m in range(0,ninds):
        file_text = file_text+\
        '    ind_arg'+str(m)+'_s = ('+indtyps[m]+' *) &shared[nbytes];\n'
        if m < ninds-1:
          file_text = file_text+\
          '    nbytes    += ROUND_UP(ind_arg'+str(m)+'_size*sizeof('+indtyps[m]+')*'+ inddims[m]+');\n'
        
      file_text = file_text+\
      '  }\n'\
      '  __syncthreads(); // make sure all of above completed\n'\
      '\n  // copy indirect datasets into shared memory or zero increment\n\n'
       
      for m in range(0,ninds):
        if indaccs[m]==OP_READ or indaccs[m]==OP_RW or indaccs[m]==OP_INC:
          file_text = file_text+\
          '  for (int n=threadIdx.x; n<ind_arg'+str(m)+'_size*'+inddims[m]+'; n+=blockDim.x)\n'
          if indaccs[m]==OP_READ or indaccs[m]==OP_RW:
            file_text = file_text+\
            '      ind_arg'+str(m)+'_s[n] = ind_arg'+str(m)+'[n%'+inddims[m]+\
            '+ind_arg'+str(m)+'_map[n/'+inddims[m]+']*'+inddims[m]+'];\n\n'
          elif indaccs[m]==OP_INC:
            file_text = file_text+\
            '      ind_arg'+str(m)+'_s[n] = ZERO_'+indtyps[m]+';\n'
       
      file_text = file_text+'\n'\
      '  __syncthreads();\n'\
      '  // process set elements\n\n'
       
      if ind_inc:
        file_text = file_text+\
        '  for (int n=threadIdx.x; n<nelems2; n+=blockDim.x) {\n'\
        '    int col2 = -1;                               \n'\
        '    if (n<nelem) {                               \n\n'\
        '      // initialise local variables            \n\n'

        for m in range(0,nargs):
          if maps[m]==OP_MAP and accs[m]==OP_INC:
            file_text = file_text+\
            '      for (int d=0; d<'+dims[m]+'; d++)\n'\
            '        arg'+str(m)+'_l[d] = ZERO_'+typs[m]+';\n'
      else:
        file_text = file_text+'    for (int n=threadIdx.x; n<nelem; n+=blockDim.x) {\n'
#
# simple alternative when no indirection
#
    else:
      use_shared = 0;
      for m in range(0,nargs):
        if maps[m]<>OP_GBL and dims[m]<>'1':
          use_shared = 1
      
      if use_shared:
        file_text = file_text+'  int   tid = threadIdx.x%OP_WARPSIZE;\n\n'\
        '  extern __shared__ char shared[];    \n'\
        '  char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);\n'
    
      file_text = file_text+\
      '\n  // process set elements\n'\
      '  for (int n=threadIdx.x+blockIdx.x*blockDim.x;\n'\
      '       n<set_size; n+=blockDim.x*gridDim.x) {\n'
      
      if use_shared:
        file_text = file_text+\
        '    int offset = n - tid;\n'\
        '    int nelems = MIN(OP_WARPSIZE,set_size-offset);\n'\
        '    // copy data into shared memory, then into local\n\n'
      
      for m in range(0,nargs):
        if (maps[m]<>OP_GBL and accs[m]<>OP_WRITE and dims[m]<>'1') and not(soaflags[m]):
          file_text = file_text+\
          '    for (int m=0; m<'+dims[m]+'; m++)\n'\
          '      (('+typs[m]+' *)arg_s)[tid+m*nelems] = arg'+str(m)+'[tid+m*nelems+offset*'+dims[m]+'];\n\n'\
          '    for (int m=0; m<'+dims[m]+'; m++)\n'\
          '      arg'+str(m)+'_l[m] = (('+typs[m]+' *)arg_s)[m+tid*'+dims[m]+'];\n'
        
    
                        
#
# kernel call#   

    # xxx: array of pointers for non-locals 
    for m in range(1,ninds+1):
      s = [i for i in range(len(inds)) if inds[i]==m]
      if sum(s)>1:
        if indaccs[m-1] <> OP_INC:
          file_text = file_text+' \n'
          ctr = 0
          for n in range(0,nargs):
            if inds[n] == m and vectorised[m]:
              file_text = file_text+\
              '      arg'+str(m-1)+'_vec['+str(ctr)+'] = ind_arg'+str(inds[n]-1)+'_s+arg_map['+\
              str(cumulative_indirect_index[n])+'*set_size+n+offset_b]*'+str(dims[n])+';\n'
              ctr = ctr+1
    
    file_text = file_text+'\n      // user-supplied kernel call\n\n'

    line = '      '+name+'('
    
    a = 0 #only apply indentation if its not the 0th argument
    indent =''
    for m in range (0, nargs):
      if a > 0:
        indent = '       '+' '*len(name)
        
      if maps[m] == OP_GBL:
        if accs[m] == OP_READ:
          line = line+indent+'arg'+str(m)+',\n'
        else:
          line = line+indent+'arg'+str(m)+'_l,\n'
        a =a+1
      elif maps[m]==OP_MAP and  accs[m]==OP_INC and vectorised[m]==0:
        line = line+indent+'arg'+str(m)+'_l,\n'
        a =a+1
      elif maps[m]==OP_MAP and vectorised[m]==0:
        line = line+indent+'ind_arg'+str(inds[m]-1)+'_s+arg_map['+\
        str(cumulative_indirect_index[m])+'*set_size+n+offset_b]*'+str(dims[m])+','+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m == 0:
        line = line+indent+'arg'+str(inds[m]-1)+'_vec,'+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] <> vectorised[m-1]: #xxx:vector
        line = line+indent+'arg'+str(inds[m]-1)+'_vec,'+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] == vectorised[m-1]:
        line = line
        a =a+1
      elif maps[m]==OP_ID:
        if ninds>0:
          if soaflags[m]:
            line = line+indent+'arg'+str(m)+'+(n+offset_b),\n'
          else:
            line = line+indent+'arg'+str(m)+'+(n+offset_b)*'+str(dims[m])+','+'\n'
          a =a+1
        else:
          if dims[m] == '1' or soaflags[m]:
            line = line+indent+'arg'+str(m)+'+n,\n'
          else:
            line = line+indent+'arg'+str(m)+'_l,\n'
          a =a+1
      else:
        print 'internal error 1 '
    
    file_text = file_text+line[0:-2]+');\n\n' #remove final ',' and \n  
    
#
# updating for indirect kernels ...
#    
    if ninds>0:
      if ind_inc:
        file_text = file_text+\
        '      col2 = colors[n+offset_b];        \n'\
        '    }\n\n\n'\
        '    // store local variables            \n\n'
      
        for m in range(0,nargs):
          if maps[m]==OP_MAP and accs[m]==OP_INC:
            file_text = file_text+'    int arg'+str(m)+'_map;\n'
        
        file_text = file_text+'    if (col2>=0) {\n'
        
        for m in range(0,nargs):
          if maps[m] == OP_MAP and accs[m] == OP_INC:
            file_text = file_text+\
            '      arg'+str(m)+'_map = arg_map['+str(cumulative_indirect_index[m])+'*set_size+n+offset_b];\n'
        
        file_text = file_text+\
        '    }\n\n'\
        '    for (int col=0; col<ncolor; col++) {\n'\
                '      if (col2==col) {'
        
        for m in range(0,nargs):
          if maps[m] == OP_MAP and accs[m] == OP_INC:
            file_text = file_text+'\n'+\
            '        for (int d=0; d<'+str(dims[m])+'; d++)\n'\
            '          ind_arg'+str(inds[m]-1)+'_s[d+arg'+str(m)+'_map*'+dims[m]+'] += arg'+str(m)+'_l[d];'
      
        file_text = file_text +'\n'+\
        '      }\n'\
        '      __syncthreads();\n'\
        '    }\n'
      file_text = file_text +\
      '  }\n'
      
      s = [i for i in range(1,ninds+1) if indaccs[i-1]<> OP_READ]
      
      if len(s)>0 and max(s)>0:
        file_text = file_text +'\n  // apply pointered write/increment\n'
        
      for m in range(1,ninds+1):
        if indaccs[m-1]==OP_WRITE or indaccs[m-1]==OP_RW or indaccs[m-1]==OP_INC:
          file_text = file_text +\
          '  for (int n=threadIdx.x; n<ind_arg'+str(m-1)+'_size*'+inddims[m-1]+'; n+=blockDim.x)\n'
          if indaccs[m-1]==OP_WRITE or indaccs[m-1]==OP_RW:
            file_text = file_text +\
            '    ind_arg'+str(m-1)+'[n%'+inddims[m-1]+'+ind_arg'+str(m-1)+\
            '_map[n/'+inddims[m-1]+']*'+inddims[m-1]+'] = ind_arg'+str(m-1)+'_s[n];'
            
          elif indaccs[m-1]==OP_INC:
            file_text = file_text +\
            '    ind_arg'+str(m-1)+'[n%'+inddims[m-1]+'+ind_arg'+str(m-1)+\
            '_map[n/'+inddims[m-1]+']*'+inddims[m-1]+'] += ind_arg'+str(m-1)+'_s[n];'

#
# ... and direct kernels
#
    else:
      if use_shared:
        file_text = file_text +\
        '    // copy back into shared memory, then to device\n'
      for m in range(0,nargs):
        if (maps[m]<>OP_GBL and accs[m]<>OP_READ and dims[m]<>'1') and not(soaflags[m]):
          file_text = file_text +'\n'+\
          '    for (int m=0; m<'+dims[m]+'; m++)\n'\
          '      (('+typs[m]+' *)arg_s)[m+tid*'+dims[m]+'] = arg'+str(m)+'_l[m];\n\n'\
          '    for (int m=0; m<'+dims[m]+'; m++)\n'\
          '      arg'+str(m)+'[tid+m*nelems+offset*'+dims[m]+'] = (('+typs[m]+' *)arg_s)[tid+m*nelems];\n'
      file_text = file_text +'  }\n'

#
# global reduction
#
    if reduct:
       file_text = file_text +'\n'\
       '  // global reductions\n\n'
       for m in range (0,nargs):
         if maps[m]==OP_GBL and accs[m]<>OP_READ:
           file_text = file_text +'  for(int d=0; d<'+dims[m]+'; d++)\n'
           if accs[m]==OP_INC:
             file_text = file_text +\
             '    op_reduction<OP_INC>(&arg'+str(m)+'[d+blockIdx.x*'+dims[m]+'],arg'+str(m)+'_l[d]);\n'
           elif accs[m]==OP_MIN:
             file_text = file_text +\
             '    op_reduction<OP_MIN>(&arg'+str(m)+'[d+blockIdx.x*'+dims[m]+'],arg'+str(m)+'_l[d]);\n'
           elif accs[m]==OP_MAX:
             file_text = file_text +\
             '    op_reduction<OP_MAX>(&arg'+str(m)+'[d+blockIdx.x*'+dims[m]+'],arg'+str(m)+'_l[d]);\n';
           else:
             print 'internal error: invalid reduction option'
             sys.exit(2);
        
    
    file_text = file_text +'\n}\n'    

##########################################################################
# then C++ stub function
##########################################################################

    file_text = file_text+'\n'\
    '// host stub function          \n'\
    'void op_par_loop_'+name+'(char const *name, op_set set,\n'
    
    for m in unique_args:
        if m == unique_args[len(unique_args)-1]:
          file_text = file_text+'  op_arg arg'+str(m-1)+'){\n\n'
        else:
          file_text = file_text+'  op_arg arg'+str(m-1)+',\n'
    
    for m in range (0,nargs):
      if maps[m]==OP_GBL:
        file_text = file_text+'  '+typs[m]+' *arg'+str(m)+'h = ('+typs[m]+' *)arg'+str(m)+'.data;\n'
    
    file_text = file_text + '\n'\
    '  int nargs = '+str(nargs)+';\n'\
    '  op_arg args['+str(nargs)+'];\n\n'
  
    #print vectorised
    
    for m in range (0,nargs):
      u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == m]
      if len(u) > 0 and vectorised[m] > 0:
        file_text = file_text +\
        '  arg'+str(m)+'.idx = 0;\n'\
        '  args['+str(m)+'] = arg'+str(m)+';\n'
        
        v = [int(vectorised[i] == vectorised[m]) for i in range(0,len(vectorised))]
        first = [i for i in range(0,len(v)) if v[i] == 1]
        first = first[0]
        file_text = file_text +\
        '  for (int v = 1; v < '+str(sum(v))+'; v++) {\n'\
        '    args['+str(m)+' + v] = op_arg_dat(arg'+str(first)+'.dat, v, arg'+\
        str(first)+'.map, '+dims[m]+', "'+typs[m]+'", '+accsstring[accs[m]-1]+');\n  }\n' 
        
      elif vectorised[m]>0:
        file_text = file_text
      else:
        file_text = file_text +'  args['+str(m)+'] = arg'+str(m)+';\n'
        
#
#   indirect bits
#
    if ninds>0:
      file_text = file_text +'\n'\
      '  int    ninds   = '+str(ninds)+';\n'\
      '  int    inds['+str(nargs)+'] = {'
      for m in range(0,nargs):
        file_text = file_text + str(inds[m]-1)+','
        
      file_text = file_text[:-1] + '};\n\n'
      
      file_text = file_text + \
      '  if (OP_diags>2) {\n'\
      '    printf(" kernel routine with indirection: '+name+'\\n");\n'\
      '  }\n\n'\
      '  // get plan\n'\
      '  #ifdef OP_PART_SIZE_'+ str(nk)+'\n'\
      '    int part_size = OP_PART_SIZE_'+str(nk)+';\n'\
      '  #else\n'\
      '    int part_size = OP_part_size;\n'\
      '  #endif\n\n'\
      '  int set_size = op_mpi_halo_exchanges(set, nargs, args);\n'  
    
#
# direct bit
#   
    else:
      file_text = file_text + \
      '\n  if (OP_diags>2) {\n'\
      '    printf(" kernel routine w/o indirection:  '+ name + '");\n'\
      '  }\n\n'\
      '  op_mpi_halo_exchanges(set, nargs, args);\n'

#
# start timing
#
    file_text = file_text + '\n  // initialise timers\n'\
                '  double cpu_t1, cpu_t2, wall_t1, wall_t2;\n'\
                '  op_timers_core(&cpu_t1, &wall_t1);\n'

    file_text = file_text +'\n  if (set->size >0) {\n\n'
    
    if sum(soaflags):
      file_text = file_text +'    int op2_stride = set->size + set->exec_size + set->nonexec_size;\n'
      file_text = file_text +'    op_decl_const_char(1, "int", sizeof(int), (char *)&op2_stride, "op2_stride");\n\n'
    
#
# kernel call for indirect version
#
    if ninds>0:
      file_text = file_text +\
      '    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);\n\n'
      

#
# transfer constants
#
    g = [i for i in range(0,nargs) if maps[i] == OP_GBL and accs[i] == OP_READ]
    if len(g)>0: 
      file_text = file_text +\
      '    // transfer constants to GPU\n'\
      '    int consts_bytes = 0;\n'
      for m in range(0,nargs):
        if maps[m]==OP_GBL and accs[m]==OP_READ:
          file_text = file_text +\
          '    consts_bytes += ROUND_UP('+dims[m]+'*sizeof('+typs[m]+'));\n'
      
      file_text = file_text +'\n'\
      '    reallocConstArrays(consts_bytes);\n\n'
      '    consts_bytes = 0;\n'

      for m in range(0,nargs):
        if maps[m]==OP_GBL and accs[m]==OP_READ:
          file_text = file_text +\
          '    arg'+str(m)+'.data   = OP_consts_h + consts_bytes;\n'\
          '    arg'+str(m)+'.data_d = OP_consts_d + consts_bytes;\n'\
          '    for (int d=0; d<'+dims[m]+'; d++) (('+typs[m]+' *)arg'+\
          str(m)+'.data)[d] = arg'+str(m)+'h[d];\n'\
          '    consts_bytes += ROUND_UP('+dims[m]+'*sizeof('+typs[m]+'));\n'\
       
      file_text = file_text +'\n'\
      '    mvConstArraysToDevice(consts_bytes);\n\n'


#
# transfer global reduction initial data
#

    if ninds == 0:
      file_text = file_text +\
      '    // set CUDA execution parameters\n\n'\
      '    #ifdef OP_BLOCK_SIZE_'+str(nk)+'\n'\
      '      int nthread = OP_BLOCK_SIZE_'+str(nk)+';\n'\
      '    #else\n'\
      '      // int nthread = OP_block_size;\n'\
      '      int nthread = 128;\n'\
      '    #endif\n\n'\
      '    int nblocks = 200;\n\n'
      
    
    if reduct:
      file_text = file_text +'    // transfer global reduction data to GPU\n\n'
      if ninds>0:
        file_text = file_text +\
           '    int maxblocks = 0;\n'\
           '    for (int col=0; col < Plan->ncolors; col++)\n'\
           '      maxblocks = MAX(maxblocks,Plan->ncolblk[col]);\n'
      else:
        file_text = file_text +\
        '    int maxblocks = nblocks;\n\n'
 
      file_text = file_text +\
      '    int reduct_bytes = 0;\n'\
      '    int reduct_size  = 0;\n'

      for m in range(0,nargs):
        if maps[m]==OP_GBL and accs[m]<>OP_READ:
          file_text = file_text +\
          '    reduct_bytes += ROUND_UP(maxblocks*'+dims[m]+'*sizeof('+typs[m]+'));\n'\
          '    reduct_size   = MAX(reduct_size,sizeof('+typs[m]+'));\n'
    
      file_text = file_text +\
      '\n    reallocReductArrays(reduct_bytes);\n\n'\
      '    reduct_bytes = 0;\n'
    
      for m in range(0,nargs):
        if maps[m]==OP_GBL and accs[m]<>OP_READ:
          file_text = file_text +\
          '    arg'+str(m)+'.data   = OP_reduct_h + reduct_bytes;\n'\
          '    arg'+str(m)+'.data_d = OP_reduct_d + reduct_bytes;\n'\
          '    for (int b=0; b<maxblocks; b++)\n'\
          '      for (int d=0; d<'+dims[m]+'; d++)\n'
          if accs[m]==OP_INC:
            file_text = file_text +\
            '        (('+typs[m]+' *)arg'+str(m)+'.data)[d+b*'+dims[m]+'] = ZERO_'+typs[m]+';\n';
          else:
            file_text = file_text +\
            '        (('+typs[m]+' *)arg'+str(m)+'.data)[d+b*'+dims[m]+'] = arg'+str(m)+'h[d];\n'
          file_text = file_text +\
          '    reduct_bytes += ROUND_UP(maxblocks*'+dims[m]+'*sizeof('+typs[m]+'));\n'
           
      file_text = file_text +'\n    mvReductArraysToDevice(reduct_bytes);\n\n'
    
#
# kernel call for indirect version
#
    if ninds>0:
      file_text = file_text +'\n'\
      '    // execute plan\n\n'\
      '    int block_offset = 0;\n\n'\
      '    for (int col=0; col < Plan->ncolors; col++) {\n\n'\
      '      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs, args);\n\n'\
      '    #ifdef OP_BLOCK_SIZE_'+str(nk)+'\n'\
      '      int nthread = OP_BLOCK_SIZE_'+str(nk)+'; \n'\
      '    #else\n'\
      '      int nthread = OP_block_size;\n'\
      '    #endif\n\n'\
      '      dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],\n'\
      '                      Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);\n'\
      '      if (Plan->ncolblk[col] > 0) {\n'
      
      if reduct:
        file_text = file_text +\
        '        int nshared = MAX(Plan->nshared,reduct_size*nthread);\n'
      else:
        file_text = file_text +\
        '        int nshared = Plan->nsharedCol[col];\n'
    
      file_text = file_text +\
      '        op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(\n'

      for m in range(1,ninds+1):
        file_text = file_text +\
        '         ('+typs[invinds[m-1]]+' *)arg'+str(invinds[m-1])+'.data_d,\n'
    
      file_text = file_text +\
      '         Plan->ind_map,\n'\
      '         Plan->loc_map,\n'

      for m in range(0,nargs):
        if inds[m]==0:
          file_text = file_text +\
          '         ('+typs[m]+' *)arg'+str(m)+'.data_d,\n'
      

      file_text = file_text+\
      '         Plan->ind_sizes,\n'\
      '         Plan->ind_offs,\n'\
      '         block_offset,\n'\
      '         Plan->blkmap,\n'\
      '         Plan->offset,\n'\
      '         Plan->nelems,\n'\
      '         Plan->nthrcol,\n'\
      '         Plan->thrcol,\n'\
      '         Plan->ncolblk[col],\n'\
      '         set_size);\n\n'\
      '         cutilSafeCall(cudaThreadSynchronize());\n'\
      '         cutilCheckMsg("op_cuda_'+name+' execution failed\\n");\n'
      if reduct:
        file_text = file_text+\
        '        // transfer global reduction data back to CPU\n'
        '        if (col == Plan->ncolors_owned)\n'
        '          mvReductArraysToHost(reduct_bytes);\n'
      
      file_text = file_text+\
      '      }\n\n'\
      '      block_offset += Plan->ncolblk[col]; \n'\
      '    }\n'    
#
# kernel call for direct version
#
    else:
      file_text = file_text +\
       '    // work out shared memory requirements per element\n\n'\
       '    int nshared = 0;\n'
       
      for m in range(0,nargs):
         if maps[m]<>OP_GBL and dims[m]<>'1':
           file_text = file_text +\
           '    nshared = MAX(nshared,sizeof('+typs[m]+')*'+dims[m]+');\n\n'
        
      file_text = file_text +'\n'\
       '    // execute plan\n\n'\
       '    int offset_s = nshared*OP_WARPSIZE;\n\n'
    
      if reduct:
        file_text = file_text +\
        '    nshared = MAX(nshared*nthread,reduct_size*nthread);\n\n'
      else:
        file_text = file_text +\
        '    nshared = nshared*nthread;\n\n'
    
      file_text = file_text +\
      '    op_cuda_'+name+'<<<nblocks,nthread,nshared>>>('
    
      indent = ' '*(len(name)+42)
      for m in range(0,nargs):
        if m > 0:
          file_text = file_text +indent+'('+typs[m]+' *) arg'+str(m)+'.data_d,\n'
        else:
          file_text = file_text +'('+typs[m]+' *) arg'+str(m)+'.data_d,\n'
        
      file_text = file_text +indent+'offset_s,\n'
      file_text = file_text +indent+'set->size );\n'
      file_text = file_text +'    cutilSafeCall(cudaThreadSynchronize());\n'\
      '    cutilCheckMsg("op_cuda_'+name+' execution failed\\n");\n'
  
    if ninds>0:
      file_text = file_text +\
      '    op_timing_realloc('+str(nk)+');\n'\
      '    OP_kernels['+str(nk)+'].transfer  += Plan->transfer; \n'\
      '    OP_kernels['+str(nk)+'].transfer2 += Plan->transfer2;\n'
    
    
#
# transfer global reduction initial data
#
    if reduct:
      if ninds == 0:
        file_text = file_text +\
        '\n    // transfer global reduction data back to CPU\n\n'\
        '    mvReductArraysToHost(reduct_bytes);\n\n'
    
      for m in range(0,nargs):
        if maps[m]==OP_GBL and accs[m]<>OP_READ:
          file_text = file_text+\
          '    for (int b=0; b<maxblocks; b++)\n'\
          '        for (int d=0; d<'+dims[m]+'; d++)\n'
          if accs[m]==OP_INC:
            file_text = file_text+\
            '          arg'+str(m)+'h[d] = arg'+str(m)+'h[d] + (('+typs[m]+' *)arg'+str(m)+'.data)[d+b*'+dims[m]+'];\n'
          elif accs[m]==OP_MIN:
            file_text = file_text +\
            '          arg'+str(m)+'h[d] = MIN(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);\n'
          elif accs[m]==OP_MAX:
            file_text = file_text+\
            '          ARGh[d] = MAX(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);\n'
            
          file_text = file_text+'\n'\
          '    arg'+str(m)+'.data = (char *)arg'+str(m)+'h;\n\n'\
          '    op_mpi_reduce(&arg'+str(m)+',arg'+str(m)+'h);\n'
      
    file_text = file_text+\
    '  }\n'\
    '\n  op_mpi_set_dirtybit(nargs, args);\n\n'

#
# update kernel record
#

    file_text = file_text+\
    '  // update kernel record\n'\
    '  op_timers_core(&cpu_t2, &wall_t2);\n'\
    '  op_timing_realloc('+str(nk)+');\n'\
    '  OP_kernels[' +str(nk)+ '].name      = name;\n'\
    '  OP_kernels[' +str(nk)+ '].count    += 1;\n'\
    '  OP_kernels[' +str(nk)+ '].time     += wall_t2 - wall_t1;\n'

    if ninds == 0:
      line = '  OP_kernels['+str(nk)+'].transfer += (float)set->size *'

      for m in range (0,nargs):
        if maps[m]<>OP_GBL:
          if accs[m]==OP_READ or accs[m]==OP_WRITE:
            file_text = file_text+line+' arg'+str(m)+'.size;\n'
          else:
            file_text = file_text+line+' arg'+str(m)+'.size * 2.0f;\n'
       

    file_text = file_text+'}'


##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_kernel.cu','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################
    
  file_text = '// header\n'\
               '#include "op_lib_cpp.h"\n'\
               '#include "op_cuda_rt_support.h"\n'\
               '#include "op_cuda_reduction.h"\n\n'\
               '// global constants\n'\
               '#ifndef MAX_CONST_SIZE\n'\
               '#define MAX_CONST_SIZE 128\n'\
               '#endif\n\n'
              
  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      file_text = file_text +\
      '__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+';\n'
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'
    
      file_text = file_text +\
      '__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'['+num+'];\n'
      
  if any_soa:
    file_text = file_text +\
    '__constant__ int op2_stride;\n\n'\
    '#define OP2_STRIDE(arr, idx) arr[op2_stride*(idx)]\n\n'
  
  file_text = file_text +\
      '\nvoid op_decl_const_char(int dim, char const *type,\n'\
      '            int size, char *dat, char const *name){\n'
   
  for nc in range(0,len(consts)):
    if consts[nc]['dim'] < 0:
      file_text = file_text +\
      '  if(~strcmp(name,"'+name+'") && size>MAX_CONST_SIZE) {\n'\
      '    printf("error: MAX_CONST_SIZE not big enough\n"); exit(1);\n'\
      '  }\n'
  
  
  file_text = file_text +\
  '  cutilSafeCall(cudaMemcpyToSymbol(name, dat, dim*size));\n'\
  '}\n\n'\
  '// user kernel files\n'

  for nk in range(0,len(kernels)):
    file_text = file_text +\
    '#include "'+kernels[nk]['name']+'_kernel.cu"\n'
  
  master = master.split('.')[0] 
  fid = open(master.split('.')[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()
  
              
    
