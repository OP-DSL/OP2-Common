##########################################################################
#
# OpenMP code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.cpp for each kernel,
# plus a master kernel file
#
##########################################################################

def op2_gen_openmp(master, date, consts, kernels):

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
    
    #print inds
    #print invinds
    
    vec =  [m for m in range(0,nargs) if int(idxs[m])<0 and maps[m] == OP_MAP]
    if len(vec) > 0:
      unique_args = 1;
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
      	    if m > 1:
      	      unique_args = [unique_args,len(new_dims)+1]
      	    
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
            if m>1:
              unique_args = [unique_args,len(new_dims)+1]
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
    	unique_args = range(0,nargs)
        
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
#  start with OpenMP kernel function
##########################################################################

    file_text = '//user function\n'\
    '#include '+name+'.h\n'\
    '\n// x86 kernel function\n\n'\
    'void op_x86_'+name+'(\n'
    if ninds>0:
      file_text = file_text+'  int    blockIdx,\n'
  
    for m in range(0,ninds):
      file_text = file_text+'  '+str(indtyps[m])+' *ind_arg'+str(m)+',\n'
    
    if ninds>0:
      file_text = file_text+'  int   *ind_map,\n'
      file_text = file_text+'  short *arg_map,\n'
  
    for m in range (0,nargs):
      if maps[m]==OP_GBL and accs[m] == OP_READ:
        file_text = file_text+'  const '+typs[m]+\
        ' *arg'+str(m)+',\n'  # declared const for performance
      elif maps[m]==OP_ID and ninds>0:
        file_text = file_text+'  '+typs[m]+ ' *arg'+str(m)+',\n'
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
      '  int   set_size) {   \n'
    else:
      file_text = file_text+'  int   start,    \n'\
      '  int   finish ) {\n'
  
  
    print file_text
