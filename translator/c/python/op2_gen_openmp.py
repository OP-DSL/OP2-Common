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
      '  int   set_size) {   \n\n'
    else:
      file_text = file_text+'  int   start,    \n'\
      '  int   finish ) {\n\n'
  
    for m in range (0,nargs):
      if maps[m]==OP_MAP and accs[m]==OP_INC:
        file_text = file_text+'  '+typs[m]+'  arg'+str(m)+'_l['+dims[m]+'];\n'
    
    for m in range (1,ninds+1):
      v = [i for i in range(len(inds)) if inds[i]==m]
      v_i = [] 
      for i in range(len(v)):
        v_i = v_i + [vectorised[v[i]]]
      if sum(v)>1 and len(v_i)>0:
        if indaccs[m-1] == OP_INC:
          ind = max([idxs[i] for i in range(len(inds)) if inds[i]==m])
          file_text = file_text+'  '+indtyps[m-1]+' *arg'+str(m-1)+\
          '_vec['+str(ind+1)+'] = {\n'
          for n in range(0,nargs):
            if inds[n] == m:
              file_text = file_text+'    arg'+str(n)+'_l,\n'
          file_text = file_text+'  };\n'
        else:
         ind = max([idxs[i] for i in range(len(inds)) if inds[i]==m])
         file_text = file_text+'  '+indtyps[m-1]+' *arg'+str(m-1)+\
         '_vec['+str(ind+1)+'];\n'
# 
# lengthy code for general case with indirection
#
    if ninds>0:
      file_text = file_text+' \n'
      for m in range (0,ninds):
        file_text = file_text+'  int  *ind_arg'+str(m)+\
        '_map, ind_arg'+str(m)+'_size;\n'
      for m in range (0,ninds):
        file_text = file_text+'  '+indtyps[m]+' *ind_arg'+str(m)+'_s;\n'
       
      file_text = file_text+\
        '  int    nelem, offset_b;\n\n'\
        '  char shared[128000];\n'\
        '  if (0==0) {\n\n'\
        '    // get sizes and shift pointers and direct-mapped data\n\n'\
        '    int blockId = blkmap[blockIdx + block_offset];\n'\
        '    nelem    = nelems[blockId];\n'\
        '    offset_b = offset[blockId];\n\n'
       
      for m in range (0,ninds):
        file_text = file_text+'    ind_arg'+str(m)+'_size = ind_arg_sizes['+\
        str(m)+'+blockId*'+ str(ninds)+'];\n'
      
      file_text = file_text+'\n'
       
      for m in range (1,ninds+1):
        c = [i for i in range(len(inds)) if inds[i]==m]
        file_text = file_text+'    ind_arg'+str(m)+'_map = &ind_map['+\
        str(cumulative_indirect_index[c[0]])+'*set_size] + ind_arg_offs['+\
        str(m-1)+'+blockId*'+str(ninds)+'];\n'
       
      file_text = file_text+'\n    // set shared memory pointers\n'\
      '    int nbytes = 0;\n'
       
      for m in range(0,ninds):
        file_text = file_text+'    ind_arg'+str(m)+'_s = ('+indtyps[m]+\
        ' *) &shared[nbytes];\n'
        if m < ninds-1:
          file_text = file_text+ '    nbytes    += ROUND_UP(ind_arg+'+\
          str(m)+'_size*sizeof('+\
          indtyps[m]+')*'+ inddims[m]+');\n'
        
      file_text = file_text+'  }\n'\
      '\n  // copy indirect datasets into shared memory or zero increment\n\n'
       
      for m in range(0,ninds):
        if indaccs[m]==OP_READ or indaccs[m]==OP_RW or indaccs[m]==OP_INC:
          file_text = file_text+ '  for (int n=0; n<ind_arg'+str(m)+'_size; n++)\n'
          file_text = file_text+ '    for (int d=0; d<'+inddims[m]+'; d++)\n'
        if indaccs[m]==OP_READ or indaccs[m]==OP_RW:
          file_text = file_text+ '      ind_arg'+str(m)+'_s[d+n*'+\
          inddims[m]+'] = ind_arg'+str(m)+\
          '[d+ind_arg'+str(m)+'_map[n]*'+inddims[m]+'];\n\n'
        elif indaccs[m]==OP_INC:
          file_text = file_text+ '      ind_arg'+str(m)+'_s[d+n*'+\
          inddims[m]+'] = ZERO_'+indtyps[m]+';\n'
       
      file_text = file_text+'\n  // process set elements\n\n'
       
      if ind_inc:
        file_text = file_text+'  for (int n=0; n<nelem; n++) {\n'\
        '    // initialise local variables            \n\n'

        for m in range(0,nargs):
          if maps[m]==OP_MAP and accs[m]==OP_INC:
            file_text = file_text+ '    for (int d=0; d<'+dims[m]+'; d++)\n'\
            '      arg'+str(m)+'_l[d] = ZERO_'+typs[m]+';\n'
      else:
        file_text = file_text+'  for (int n=0; n<nelem; n++) {\n'
#
# simple alternative when no indirection
#
    else:
      file_text = file_text+'  // process set elements\n'\
                        '  for (int n=start; n<finish; n++) {\n'    
#
# kernel call
#   

    # xxx: array of pointers for non-locals 
    for m in range(1,ninds+1):
      s = [i for i in range(len(inds)) if inds[i]==m]
      if sum(s)>1:
        if indaccs[m-1] <> OP_INC:
          file_text = file_text+' \n'
          ctr = 0
          for n in range(0,nargs):
            if inds[n] == m and vectorised[m]:
              file_text = file_text+'    arg'+str(m)+\
              '_vec['+str(ctr)+'] = ind_arg'+str(inds[n]-1)+'_s+arg_map['+\
              str(cumulative_indirect_index[n])+'*set_size+n+offset_b]*'+str(dims[n])+';\n'
              ctr = ctr+1
    
    file_text = file_text+'\n    // user-supplied kernel call\n\n'

    line = '    '+name+'('
    prefix = ' '*len(name)
    a = 0 #only apply indentation if its not the 0th argument
    indent =''
    for m in range (0, nargs):
      if a > 0:
        indent = '     '+' '*len(name)
        
      if maps[m] == OP_GBL:
        line = line+indent+'arg'+str(m)+',\n'
        a =a+1
      elif maps[m]==OP_MAP and  accs[m]==OP_INC and vectorised[m]==0:
        line = line+indent+'arg_l'+str(m)+',\n'
        a =a+1
      elif maps[m]==OP_MAP and vectorised[m]==0:
        line = line+indent+'ind_arg'+str(inds[m]-1)+'_s+arg_map['+\
        str(cumulative_indirect_index[m])+'*set_size+n+offset_b]*'+str(m)+','+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m == 1:
        line = line+indent+'arg'+str(inds[m]-1)+'_vec,'+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m>1 and vectorised[m] <> vectorised[m-1]: #xxx:vector
        line = line+indent+'arg'+str(inds[m]-1)+'_vec,'+'\n'
        a =a+1
      elif maps[m]==OP_MAP and m>1 and vectorised[m] == vectorised[m-1]:
        line = line
        a =a+1
      elif maps[m]==OP_ID:
        if ninds>0:
          line = line+indent+'arg'+str(m)+'+(n+offset_b)*'+str(dims[m])+','+'\n'
          a =a+1
        else:
          line = line+indent+'arg'+str(m)+'+n*'+str(dims[m])+','+'\n'
          a =a+1
      else:
        print 'internal error 1'
    
    file_text = file_text+line[0:-2]+');\n' #remove final ',' and \n  
    
#
# updating for indirect kernels ...
#    
    if ninds>0:
      if ind_inc:
        file_text = file_text+'\n    // store local variables            \n\n'
      
        for m in range(0,nargs):
          if maps[m] == OP_MAP and accs[m] == OP_INC:
            file_text = file_text+'    int arg'+str(m)+'_map = arg_map['+\
            str(cumulative_indirect_index[m])+'*set_size+n+offset_b];\n'
      
        for m in range(0,nargs):
          if maps[m] == OP_MAP and accs[m] == OP_INC:
            file_text = file_text+'\n    for (int d=0; d<'+str(dims[m])+'; d++)\n'\
            '      ind_arg'+str(inds[m]-1)+'_s[d+arg'+str(m)+'_map*'+dims[m]+\
            '] += arg'+str(m)+'_l[d];\n'
          
      file_text = file_text + '  }\n'
      
      s = [i for i in range(1,ninds+1) if indaccs[i-1]<> OP_READ]
      if max(s)>0:
        file_text = file_text +'\n  // apply pointered write/increment\n'
        
      for m in range(1,ninds+1):
        if indaccs[m-1]==OP_WRITE or indaccs[m-1]==OP_RW or indaccs[m-1]==OP_INC:
          file_text = file_text +'  for (int n=0; n<ind_arg'+str(m-1)+'_size; n++)\n'\
          '    for (int d=0; d<'+inddims[m-1]+'; d++)\n'
          if indaccs[m-1]==OP_WRITE or indaccs[m-1]==OP_RW:
            file_text = file_text +'      ind_arg'+str(m-1)+'[d+ind_arg'+str(m-1)+'_map[n]*'+\
            inddims[m-1]+'] = ind_arg'+str(m-1)+'_s[d+n*'+inddims[m-1]+'];'
          elif indaccs[m-1]==OP_INC:
            file_text = file_text +'      ind_arg'+str(m-1)+'[d+ind_arg'+str(m-1)+'_map[n]*'+\
            inddims[m-1]+'] += ind_arg'+str(m-1)+'_s[d+n*'+inddims[m-1]+'];'
#
# ... and direct kernels
#
    else:
      file_text = file_text +'  }\n'

#
# global reduction
#
    file_text = file_text +'}\n'
    

##########################################################################
# then C++ stub function
##########################################################################

    file_text = file_text+'\n// host stub function          \n'\
        'void op_par_loop_'+name+'(char const *name, op_set set,\n'
        
    for m in range(0,len(unique_args)):
      if m == unique_args[len(unique_args)-1]:
        file_text = file_text+'  op_arg arg'+str(m)+'){\n'
      else:
        file_text = file_text+'  op_arg arg'+str(m)+',\n'
    
    print file_text
    
    #for m in range(len([i for i in range(len(unique_args)) if m == unique_args[i]])):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
