
//
// routines below are needed for generated CUDA and OpenMP code
//

//
// timing routine from Gihan Mudalige
//

#include <sys/time.h>


#include "op_rt_support.h"


//
// Global variables
//
int OP_plan_index=0, OP_plan_max=0;

op_plan   *OP_plans;



void op_rt_exit ()
{
  // free storage for plans

  for(int ip=0; ip<OP_plan_index; ip++) {
    free(OP_plans[ip].dats);
    free(OP_plans[ip].idxs);
    free(OP_plans[ip].maps);
    free(OP_plans[ip].accs);
    free(OP_plans[ip].ind_maps);
    free(OP_plans[ip].nindirect);
    free(OP_plans[ip].loc_maps);
    free(OP_plans[ip].ncolblk);
  }

  OP_plan_index=0; OP_plan_max=0;
  
  free(OP_plans);
}
  

void op_timers(double *cpu, double *et) {
  struct timeval t;
  
  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;
}

//
// comparison function for integer quicksort in op_plan
//

int comp(const void *a2, const void *b2) {
  int *a = (int *)a2;
  int *b = (int *)b2;
  
  if (*a == *b)
  return 0;
  else
  if (*a < *b)
  return -1;
  else
  return 1;
}

//
// plan check routine
//

void op_plan_check(op_plan OP_plan, int ninds, int *inds) {
  
  int err, ntot;
  
  int nblock = 0;
  for (int col=0; col<OP_plan.ncolors; col++) nblock += OP_plan.ncolblk[col];
  
  //
  // check total size
  //
  
  int nelem = 0;
  for (int n=0; n<nblock; n++) nelem += OP_plan.nelems[n];
  
  if (nelem != OP_plan.set->size) {
    printf(" *** OP_plan_check: nelems error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: nelems   OK \n");
  }
  
  //
  // check offset and nelems are consistent
  //
  
  err  = 0;
  ntot = 0;
  
  for (int n=0; n<nblock; n++) {
    err  += (OP_plan.offset[n] != ntot);
    ntot +=  OP_plan.nelems[n];
  }
  
  if (err != 0) {
    printf(" *** OP_plan_check: offset error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: offset   OK \n");
  }
  
  //
  // check blkmap permutation
  //
  
  int *blkmap = (int *) malloc(nblock*sizeof(int));
  for (int n=0; n<nblock; n++) blkmap[n] = OP_plan.blkmap[n];
  qsort(blkmap,nblock,sizeof(int),comp);
  
  err = 0;
  for (int n=0; n<nblock; n++) err += (blkmap[n] != n);
  
  free(blkmap);
  
  if (err != 0) {
    printf(" *** OP_plan_check: blkmap error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: blkmap   OK \n");
  }
  
  //
  // check ind_offs and ind_sizes are consistent
  //
  
  err  = 0;
  
  for (int i = 0; i<ninds; i++) {
    ntot = 0;
    
    for (int n=0; n<nblock; n++) {
      err  += (OP_plan.ind_offs[i+n*ninds] != ntot);
      ntot +=  OP_plan.ind_sizes[i+n*ninds];
    }
  }
  
  if (err != 0) {
    printf(" *** OP_plan_check: ind_offs error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: ind_offs OK \n");
  }
  
  //
  // check ind_maps correctly ordered within each block
  // and indices within range
  //
  
  err = 0;
  
  for (int m = 0; m<ninds; m++) {
    int m2 = 0;
    while(inds[m2]!=m) m2++;
    int set_size = OP_plan.maps[m2]->to->size;
    
    ntot = 0;
    
    for (int n=0; n<nblock; n++) {
      int last = -1;
      for (int e=ntot; e<ntot+OP_plan.ind_sizes[m+n*ninds]; e++) {
        err  += (OP_plan.ind_maps[m][e] <= last);
        last  = OP_plan.ind_maps[m][e]; 
      }
      err  += (last >= set_size);
      ntot +=  OP_plan.ind_sizes[m+n*ninds];
    }
  }
  
  if (err != 0) {
    printf(" *** OP_plan_check: ind_maps error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: ind_maps OK \n");
  }
  
  //
  // check maps (most likely source of errors)
  //
  
  err = 0;
  
  for (int m=0; m<OP_plan.nargs; m++) {
    if (OP_plan.maps[m]!=NULL) {
      op_map map = OP_plan.maps[m];
      int    m2  = inds[m];
      
      ntot = 0;
      for (int n=0; n<nblock; n++) {
        for (int e=ntot; e<ntot+OP_plan.nelems[n]; e++) {
          int p_local  = OP_plan.loc_maps[m][e];
          int p_global = OP_plan.ind_maps[m2][p_local+OP_plan.ind_offs[m2+n*ninds]]; 
          
          err += (p_global != map->map[OP_plan.idxs[m]+e*map->dim]);
        }
        ntot +=  OP_plan.nelems[n];
      }
    }
  }
  
  if (err != 0) {
    printf(" *** OP_plan_check: maps error \n");
  }
  else if (OP_diags>6) {
    printf(" *** OP_plan_check: maps     OK \n");
  }
  
  
  //
  // check thread and block coloring
  //
  
  return;
}


//
// OP plan construction
//

op_plan * op_plan_core(char const *name, op_set set, int part_size,
                       int nargs, op_arg *args, int ninds, int *inds){
  
  // first look for an existing execution plan
  
  int ip=0, match=0;
  
  while (match==0 && ip<OP_plan_index) {
    if ( (strcmp(name,        OP_plans[ip].name) == 0)
        && (set       == OP_plans[ip].set       )
        && (nargs     == OP_plans[ip].nargs     )
        && (ninds     == OP_plans[ip].ninds     )
        && (part_size == OP_plans[ip].part_size ) ) {
      match = 1;
      for (int m=0; m<nargs; m++) {
        match = match && (args[m].dat == OP_plans[ip].dats[m])
        && (args[m].map == OP_plans[ip].maps[m])
        && (args[m].idx == OP_plans[ip].idxs[m])
        && (args[m].acc == OP_plans[ip].accs[m]);
      }
    }
    ip++;
  }
  
  if (match) {
    ip--;
    if (OP_diags > 3) printf(" old execution plan #%d\n",ip);
    OP_plans[ip].count++;
    return &(OP_plans[ip]);
  }
  else {
    if (OP_diags > 1) printf(" new execution plan #%d for kernel %s\n",ip,name);
  }
  
  // work out worst case shared memory requirement per element
  
  int maxbytes = 0;
  for (int m=0; m<nargs; m++) {
    if (inds[m]>=0) maxbytes += args[m].dat->size;
  }
  
  // set blocksize and number of blocks;
  // adaptive size based on 48kB of shared memory
  
  int bsize   = part_size;   // blocksize
  if (bsize==0) bsize = (48*1024/(64*maxbytes))*64;
  int nblocks = (set->size-1)/bsize + 1;
  
  // enlarge OP_plans array if needed
  
  if (ip==OP_plan_max) {
    // printf("allocating more memory for OP_plans \n");
    OP_plan_max += 10;
    OP_plans = (op_plan *) realloc(OP_plans,OP_plan_max*sizeof(op_plan));
    if (OP_plans==NULL) {
      printf(" op_plan error -- error reallocating memory for OP_plans\n");
      exit(-1);  
    }
  }
  
  // allocate memory for new execution plan and store input arguments
  
  OP_plans[ip].dats      = (op_dat *)malloc(nargs*sizeof(op_dat));
  OP_plans[ip].idxs      = (int *)malloc(nargs*sizeof(int));
  OP_plans[ip].maps      = (op_map *)malloc(nargs*sizeof(op_map));
  OP_plans[ip].accs      = (op_access *)malloc(nargs*sizeof(op_access));
  
  OP_plans[ip].nthrcol   = (int *)malloc(nblocks*sizeof(int));
  OP_plans[ip].thrcol    = (int *)malloc(set->size*sizeof(int));
  OP_plans[ip].offset    = (int *)malloc(nblocks*sizeof(int));
  OP_plans[ip].ind_maps  = (int **)malloc(ninds*sizeof(int *));
  OP_plans[ip].ind_offs  = (int *)malloc(nblocks*ninds*sizeof(int));
  OP_plans[ip].ind_sizes = (int *)malloc(nblocks*ninds*sizeof(int));
  OP_plans[ip].nindirect = (int *)calloc(ninds,sizeof(int));
  OP_plans[ip].loc_maps  = (short **)malloc(nargs*sizeof(short *));
  OP_plans[ip].nelems    = (int *)malloc(nblocks*sizeof(int));
  OP_plans[ip].ncolblk   = (int *)calloc(set->size,sizeof(int)); // max possibly needed
  OP_plans[ip].blkmap    = (int *)calloc(nblocks,sizeof(int));
  
  for (int m=0; m<ninds; m++) {
    int count = 0;
    for (int m2=0; m2<nargs; m2++)
    if (inds[m2]==m) count++;
    OP_plans[ip].ind_maps[m] = (int *)malloc(count*set->size*sizeof(int));
  }
  
  for (int m=0; m<nargs; m++) {
    if (inds[m]>=0)
    OP_plans[ip].loc_maps[m] = (short *)malloc(set->size*sizeof(short));
    else
    OP_plans[ip].loc_maps[m] = NULL;
    
    OP_plans[ip].dats[m] = args[m].dat;
    OP_plans[ip].idxs[m] = args[m].idx;
    OP_plans[ip].maps[m] = args[m].map;
    OP_plans[ip].accs[m] = args[m].acc;
  }
  
  OP_plans[ip].name      = name;
  OP_plans[ip].set       = set;
  OP_plans[ip].nargs     = nargs;
  OP_plans[ip].ninds     = ninds;
  OP_plans[ip].part_size = part_size;
  OP_plans[ip].nblocks   = nblocks;
  OP_plans[ip].count     = 1;
  
  OP_plan_index++;
  
  // define aliases
  
  op_dat    *dats = OP_plans[ip].dats;
  int       *idxs = OP_plans[ip].idxs;
  op_map    *maps = OP_plans[ip].maps;
  op_access *accs = OP_plans[ip].accs;
  
  int  *offset    = OP_plans[ip].offset;
  int  *nelems    = OP_plans[ip].nelems;
  int **ind_maps  = OP_plans[ip].ind_maps;
  int  *ind_offs  = OP_plans[ip].ind_offs;
  int  *ind_sizes = OP_plans[ip].ind_sizes;
  int  *nindirect = OP_plans[ip].nindirect;
  
  // allocate working arrays
  
  uint **work;
  work = (uint **)malloc(ninds*sizeof(uint *));
  
  for (int m=0; m<ninds; m++) {
    int m2 = 0;
    while(inds[m2]!=m) m2++;
    
    work[m] = (uint *)malloc((maps[m2]->to)->size*sizeof(uint));
  }
  
  int *work2;
  work2 = (int *)malloc(nargs*bsize*sizeof(int));  // max possibly needed
  
  // process set one block at a time
  
  float total_colors = 0;
  
  for (int b=0; b<nblocks; b++) {
    int bs = MIN(bsize, set->size - b*bsize);
    
    offset[b] = b*bsize;    // offset for block
    nelems[b] = bs;         // size of block
    
    // loop over indirection sets
    
    for (int m=0; m<ninds; m++) {
      
      // build the list of elements indirectly referenced in this block
      
      int ne = 0;  // number of elements
      for (int m2=0; m2<nargs; m2++) {
        if (inds[m2]==m) {
          for (int e=b*bsize; e<b*bsize+bs; e++)
          work2[ne++] = maps[m2]->map[idxs[m2]+e*maps[m2]->dim];
        }
      }
      
      // sort them, then eliminate duplicates
      
      qsort(work2,ne,sizeof(int),comp);
      
      int e = 0;
      int p = 0;
      while (p<ne) {
        work2[e] = work2[p];
        while (p<ne && work2[p]==work2[e]) p++;
        e++;
      }
      ne = e;  // number of distinct elements
      
      /*
       if (OP_diags > 5) {
       printf(" indirection set %d: ",m);
       for (int e=0; e<ne; e++) printf(" %d",work2[e]);
       printf(" \n");
       }
       */
      
      
      // store mapping and renumbered mappings in execution plan
      
      for (int e=0; e<ne; e++) {
        ind_maps[m][nindirect[m]++] = work2[e];
        work[m][work2[e]] = e;   // inverse mapping
      }
      
      for (int m2=0; m2<nargs; m2++) {
        if (inds[m2]==m) {
          for (int e=b*bsize; e<b*bsize+bs; e++)
          OP_plans[ip].loc_maps[m2][e] =
          work[m][maps[m2]->map[idxs[m2]+e*maps[m2]->dim]];
        }
      }
      
      if (b==0) {
        ind_offs[m+b*ninds]  = 0;
        ind_sizes[m+b*ninds] = nindirect[m];
      }
      else {
        ind_offs[m+b*ninds]  = ind_offs[m+(b-1)*ninds]
        + ind_sizes[m+(b-1)*ninds];
        ind_sizes[m+b*ninds] = nindirect[m]
        - ind_offs[m+b*ninds];
      }
    }
    
    // now colour main set elements
    
    for (int e=b*bsize; e<b*bsize+bs; e++) OP_plans[ip].thrcol[e]=-1;
    
    int repeat  = 1;
    int ncolor  = 0;
    int ncolors = 0;
    
    while (repeat) {
      repeat = 0;
      
      for (int m=0; m<nargs; m++) {
        if (inds[m]>=0)
        for (int e=b*bsize; e<b*bsize+bs; e++)
        work[inds[m]][maps[m]->map[idxs[m]+e*maps[m]->dim]] = 0;  // zero out color array
      }
      
      for (int e=b*bsize; e<b*bsize+bs; e++) {
        if (OP_plans[ip].thrcol[e] == -1) {
          int mask = 0;
          for (int m=0; m<nargs; m++)
          if (inds[m]>=0 && accs[m]==OP_INC)
          mask |= work[inds[m]][maps[m]->map[idxs[m]+e*maps[m]->dim]]; // set bits of mask
          
          int color = ffs(~mask) - 1;   // find first bit not set
          if (color==-1) {              // run out of colors on this pass
            repeat = 1;
          }
          else {
            OP_plans[ip].thrcol[e] = ncolor+color;
            mask    = 1 << color;
            ncolors = MAX(ncolors, ncolor+color+1);
            
            for (int m=0; m<nargs; m++)
            if (inds[m]>=0 && accs[m]==OP_INC)
            work[inds[m]][maps[m]->map[idxs[m]+e*maps[m]->dim]] |= mask; // set color bit
          }
        }
      }
      
      ncolor += 32;   // increment base level
    }
    
    OP_plans[ip].nthrcol[b] = ncolors;  // number of thread colors in this block
    total_colors += ncolors;
    
    // if(ncolors>1) printf(" number of colors in this block = %d \n",ncolors);
    
    // reorder elements by color?
    
  }
  
  
  // colour the blocks, after initialising colors to 0
  
  int *blk_col;
  blk_col = (int *) malloc(nblocks*sizeof(int));
  for (int b=0; b<nblocks; b++) blk_col[b] = -1;
  
  int repeat  = 1;
  int ncolor  = 0;
  int ncolors = 0;
  
  while (repeat) {
    repeat = 0;
    
    for (int m=0; m<nargs; m++) {
      if (inds[m]>=0) 
      for (int e=0; e<(maps[m]->to)->size; e++)
      work[inds[m]][e] = 0;               // zero out color arrays
    }
    
    for (int b=0; b<nblocks; b++) {
      if (blk_col[b] == -1) {          // color not yet assigned to block
        int  bs   = MIN(bsize, set->size - b*bsize);
        uint mask = 0;
        
        for (int m=0; m<nargs; m++) {
          if (inds[m]>=0 && accs[m]==OP_INC) 
          for (int e=b*bsize; e<b*bsize+bs; e++)
          mask |= work[inds[m]][maps[m]->map[idxs[m]+e*maps[m]->dim]]; // set bits of mask
        }
        
        int color = ffs(~mask) - 1;   // find first bit not set
        if (color==-1) {              // run out of colors on this pass
          repeat = 1;
        }
        else {
          blk_col[b] = ncolor + color;
          mask    = 1 << color;
          ncolors = MAX(ncolors, ncolor+color+1);
          
          for (int m=0; m<nargs; m++) {
            if (inds[m]>=0 && accs[m]==OP_INC)
            for (int e=b*bsize; e<b*bsize+bs; e++)
            work[inds[m]][maps[m]->map[idxs[m]+e*maps[m]->dim]] |= mask;
          }
        }
      }
    }
    
    ncolor += 32;   // increment base level
  }
  
  // store block mapping and number of blocks per color
  
  
  OP_plans[ip].ncolors = ncolors;
  
  for (int b=0; b<nblocks; b++)
  OP_plans[ip].ncolblk[blk_col[b]]++;  // number of blocks of each color
  
  for (int c=1; c<ncolors; c++)
  OP_plans[ip].ncolblk[c] += OP_plans[ip].ncolblk[c-1]; // cumsum
  
  for (int c=0; c<ncolors; c++) work2[c]=0;
  
  for (int b=0; b<nblocks; b++) {
    int c  = blk_col[b];
    int b2 = work2[c];     // number of preceding blocks of this color
    if (c>0) b2 += OP_plans[ip].ncolblk[c-1];  // plus previous colors
    
    OP_plans[ip].blkmap[b2] = b;
    
    work2[c]++;            // increment counter
  }
  
  for (int c=ncolors-1; c>0; c--)
  OP_plans[ip].ncolblk[c] -= OP_plans[ip].ncolblk[c-1]; // undo cumsum
  
  // reorder blocks by color?
  
  
  // work out shared memory requirements
  
  OP_plans[ip].nshared = 0;
  float total_shared = 0;
  
  for (int b=0; b<nblocks; b++) {
    int nbytes = 0;
    for (int m=0; m<ninds; m++) {
      int m2 = 0;
      while(inds[m2]!=m) m2++;
      
      nbytes += ROUND_UP(ind_sizes[m+b*ninds]*dats[m2]->size);
    }
    OP_plans[ip].nshared = MAX(OP_plans[ip].nshared,nbytes);
    total_shared += nbytes;
  }
  
  // work out total bandwidth requirements
  
  OP_plans[ip].transfer  = 0;
  OP_plans[ip].transfer2 = 0;
  float transfer3 = 0;
  
  for (int b=0; b<nblocks; b++) {
    for (int m=0; m<nargs; m++) {
      if (inds[m]<0) {
        float fac = 2.0f;
        if (accs[m]==OP_READ) fac = 1.0f;
        if (dats[m]!=NULL) {
          OP_plans[ip].transfer  += fac*nelems[b]*dats[m]->size;
          OP_plans[ip].transfer2 += fac*nelems[b]*dats[m]->size;
          transfer3              += fac*nelems[b]*dats[m]->size; 
        }
      }
      else {
        OP_plans[ip].transfer  += nelems[b]*sizeof(short);
        OP_plans[ip].transfer2 += nelems[b]*sizeof(short);
        transfer3              += nelems[b]*sizeof(short);
      }
    }
    for (int m=0; m<ninds; m++) {
      int m2 = 0;
      while(inds[m2]!=m) m2++;
      float fac = 2.0f;
      if (accs[m2]==OP_READ) fac = 1.0f;
      OP_plans[ip].transfer +=
      fac*ind_sizes[m+b*ninds]*dats[m2]->size;
      
      // work out how many cache lines are used by indirect addressing
      
      int i_map, l_new, l_old;
      int e0 = ind_offs[m+b*ninds];
      int e1 = e0 + ind_sizes[m+b*ninds];
      
      l_old = -1;
      
      for (int e=e0; e<e1; e++) {
        i_map = ind_maps[m][e];
        l_new = (i_map*dats[m2]->size)/OP_cache_line_size;
        if (l_new>l_old) OP_plans[ip].transfer2 += fac*OP_cache_line_size;
        l_old = l_new;
        l_new = ((i_map+1)*dats[m2]->size-1)/OP_cache_line_size;
        OP_plans[ip].transfer2 += fac*(l_new-l_old)*OP_cache_line_size;
        l_old = l_new;
      }
      
      l_old = -1;
      
      for (int e=e0; e<e1; e++) {
        i_map = ind_maps[m][e];
        l_new = (i_map*dats[m2]->size)/(dats[m2]->dim*OP_cache_line_size);
        if (l_new>l_old) transfer3 += fac*dats[m2]->dim*OP_cache_line_size;
        l_old = l_new;
        l_new = ((i_map+1)*dats[m2]->size-1)/(dats[m2]->dim*OP_cache_line_size);
        transfer3 += fac*(l_new-l_old)*dats[m2]->dim*OP_cache_line_size;
        l_old = l_new;
      }
      
      // also include mappings to load/store data
      
      fac = 1.0f;
      if (accs[m2]==OP_RW) fac = 2.0f;
      OP_plans[ip].transfer  += fac*ind_sizes[m+b*ninds]*sizeof(int);
      OP_plans[ip].transfer2 += fac*ind_sizes[m+b*ninds]*sizeof(int);
      transfer3              += fac*ind_sizes[m+b*ninds]*sizeof(int);
    }
  }
  
  // print out useful information
  
  if (OP_diags>1) {
    printf(" number of blocks       = %d \n",nblocks);
    printf(" number of block colors = %d \n",OP_plans[ip].ncolors);
    printf(" maximum block size     = %d \n",bsize);
    printf(" average thread colors  = %.2f \n",total_colors/nblocks);
    printf(" shared memory required = %.2f KB \n",OP_plans[ip].nshared/1024.0f);
    printf(" average data reuse     = %.2f \n",maxbytes*(set->size/total_shared));
    printf(" data transfer (used)   = %.2f MB \n",
           OP_plans[ip].transfer/(1024.0f*1024.0f));
    printf(" data transfer (total)  = %.2f MB \n",
           OP_plans[ip].transfer2/(1024.0f*1024.0f));
    printf(" SoA/AoS transfer ratio = %.2f \n\n",
           transfer3 / OP_plans[ip].transfer2);
  }
  
  // validate plan info
  
  op_plan_check(OP_plans[ip],ninds,inds);
  
  // free work arrays
  
  for (int m=0; m<ninds; m++) free(work[m]);
  free(work);
  free(work2);
  free(blk_col);
  
  // return pointer to plan
  
  return &(OP_plans[ip]);
}

