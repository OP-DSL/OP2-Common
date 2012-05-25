#include "op_lib_cpp.h"
#include "op_lib_mat.h"
#include "op_cuda_rt_support.h"
int threadsPerBlockSize_L_0 = 512;
int setPartitionSize_L_0 = 0;
int threadsPerBlockSize_a_0 = 512;
int setPartitionSize_a_0 = 0;

__device__ void ReductionDouble8(double *volatile reductionResult,double inputValue,int reductionOperation)
{
  extern __device__ __shared__ double sharedDouble8[];
  double *volatile volatileSharedDouble8;
  int i1;
  int threadID;
  threadID = threadIdx.x;
  i1 = blockDim.x >> 1;
  __syncthreads();
  sharedDouble8[threadID] = inputValue;
  for (; i1 > warpSize; i1 >>= 1) {
    __syncthreads();
    if (threadID < i1) {
      switch(reductionOperation){
        case 0:
{
          sharedDouble8[threadID] = sharedDouble8[threadID] + sharedDouble8[threadID + i1];
          break; 
        }
        case 1:
{
          if (sharedDouble8[threadID + i1] < sharedDouble8[threadID]) {
            sharedDouble8[threadID] = sharedDouble8[threadID + i1];
          }
          break; 
        }
        case 2:
{
          if (sharedDouble8[threadID + i1] > sharedDouble8[threadID]) {
            sharedDouble8[threadID] = sharedDouble8[threadID + i1];
          }
          break; 
        }
      }
    }
  }
  __syncthreads();
  volatileSharedDouble8 = sharedDouble8;
  if (threadID < warpSize) {
    for (; i1 > 0; i1 >>= 1) {
      if (threadID < i1) {
        switch(reductionOperation){
          case 0:
{
            volatileSharedDouble8[threadID] = volatileSharedDouble8[threadID] + volatileSharedDouble8[threadID + i1];
            break; 
          }
          case 1:
{
            if (volatileSharedDouble8[threadID + i1] < volatileSharedDouble8[threadID]) {
              volatileSharedDouble8[threadID] = volatileSharedDouble8[threadID + i1];
            }
            break; 
          }
          case 2:
{
            if (volatileSharedDouble8[threadID + i1] > volatileSharedDouble8[threadID]) {
              volatileSharedDouble8[threadID] = volatileSharedDouble8[threadID + i1];
            }
            break; 
          }
        }
      }
    }
  }
  if (threadID == 0) {
    switch(reductionOperation){
      case 0:
{
         *reductionResult =  *reductionResult + volatileSharedDouble8[0];
        break; 
      }
      case 1:
{
        if (sharedDouble8[0] <  *reductionResult) {
           *reductionResult = volatileSharedDouble8[0];
        }
        break; 
      }
      case 2:
{
        if (sharedDouble8[0] >  *reductionResult) {
           *reductionResult = volatileSharedDouble8[0];
        }
        break; 
      }
    }
  }
}

__device__ void L_0_modified(double **localTensor,double *dt,double *(c0)[2UL],double *c1[1UL])
{
  const double CG1[3UL][6UL] = {{(0.0915762135097707), (0.0915762135097707), (0.8168475729804585), (0.4459484909159649), (0.4459484909159649), (0.1081030181680702)}, {(0.0915762135097707), (0.8168475729804585), (0.0915762135097707), (0.4459484909159649), (0.1081030181680702), (0.4459484909159649)}, {(0.8168475729804585), (0.0915762135097707), (0.0915762135097707), (0.1081030181680702), (0.4459484909159649), (0.4459484909159649)}};
  const double d_CG1[3UL][6UL][2UL] = {{{(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}}, {{(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}}, {{((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}}};
  const double w[6UL] = {(0.0549758718276609), (0.0549758718276609), (0.0549758718276609), (0.1116907948390057), (0.1116907948390057), (0.1116907948390057)};
  double c_q1[6UL];
  double c_q0[6UL][2UL][2UL];
  for (int i_g = 0; i_g < 6; i_g++) {
    c_q1[i_g] = 0.0;
    for (int q_r_0 = 0; q_r_0 < 3; q_r_0++) {
      c_q1[i_g] += (c1[q_r_0][0] * CG1[q_r_0][i_g]);
    }
    for (int i_d_0 = 0; i_d_0 < 2; i_d_0++) {
      for (int i_d_1 = 0; i_d_1 < 2; i_d_1++) {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for (int q_r_0 = 0; q_r_0 < 3; q_r_0++) {
          c_q0[i_g][i_d_0][i_d_1] += (c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1]);
        }
      }
    }
  }
  for (int i_r_0 = 0; i_r_0 < 3; i_r_0++) {
    for (int i_g = 0; i_g < 6; i_g++) {
      double ST1 = 0.0;
      ST1 += ((CG1[i_r_0][i_g] * c_q1[i_g]) * ((c_q0[i_g][0][0] * c_q0[i_g][1][1]) + (((-1) * c_q0[i_g][0][1]) * c_q0[i_g][1][0])));
      localTensor[i_r_0][0] += (ST1 * w[i_g]);
    }
  }
}

__global__ void L_0_kernel(double *opDat1,double *reductionArrayDevice2,double *opDat3,double *opDat4,int *ind_maps1,int *ind_maps3,int *ind_maps4,short *mappingArray1,short *mappingArray2,short *mappingArray3,short *mappingArray4,short *mappingArray5,short *mappingArray6,short *mappingArray7,short *mappingArray8,short *mappingArray9,int *pindSizes,int *pindOffs,int *pblkMap,int *poffset,int *pnelems,int *pnthrcol,int *pthrcol,int blockOffset)
{
  double opDat1Local1[1];
  double opDat1Local2[1];
  double opDat1Local3[1];
  extern __device__ __shared__ char shared_L_0[];
  __device__ __shared__ int sharedMemoryOffset;
  __device__ __shared__ int numberOfActiveThreads;
  __device__ __shared__ int nbytes;
  int blockID;
  int i1;
  double opDat2Local[1];
  __device__ __shared__ int *opDat1IndirectionMap;
  double *opDat1vec[3];
  __device__ __shared__ int *opDat3IndirectionMap;
  double *opDat3vec[3];
  __device__ __shared__ int *opDat4IndirectionMap;
  double *opDat4vec[3];
  __device__ __shared__ int opDat1SharedIndirectionSize;
  __device__ __shared__ int opDat3SharedIndirectionSize;
  __device__ __shared__ int opDat4SharedIndirectionSize;
  __device__ __shared__ double *opDat1SharedIndirection;
  __device__ __shared__ double *opDat3SharedIndirection;
  __device__ __shared__ double *opDat4SharedIndirection;
  __device__ __shared__ int numOfColours;
  __device__ __shared__ int numberOfActiveThreadsCeiling;
  int colour1;
  int colour2;
  int i2;
  for (i1 = 0; i1 < 1; ++i1) {
    opDat2Local[i1] = 0.00000;
  }
  if (threadIdx.x == 0) {
    blockID = pblkMap[blockIdx.x + blockOffset];
    numberOfActiveThreads = pnelems[blockID];
    numberOfActiveThreadsCeiling = blockDim.x * (1 + (numberOfActiveThreads - 1) / blockDim.x);
    numOfColours = pnthrcol[blockID];
    sharedMemoryOffset = poffset[blockID];
    opDat1SharedIndirectionSize = pindSizes[0 + blockID * 3];
    opDat3SharedIndirectionSize = pindSizes[1 + blockID * 3];
    opDat4SharedIndirectionSize = pindSizes[2 + blockID * 3];
    opDat1IndirectionMap = ind_maps1 + pindOffs[0 + blockID * 3];
    opDat3IndirectionMap = ind_maps3 + pindOffs[1 + blockID * 3];
    opDat4IndirectionMap = ind_maps4 + pindOffs[2 + blockID * 3];
    nbytes = 0;
    opDat1SharedIndirection = ((double *)(&shared_L_0[nbytes]));
    nbytes += ROUND_UP(opDat1SharedIndirectionSize * (sizeof(double ) * 1));
    opDat3SharedIndirection = ((double *)(&shared_L_0[nbytes]));
    nbytes += ROUND_UP(opDat3SharedIndirectionSize * (sizeof(double ) * 2));
    opDat4SharedIndirection = ((double *)(&shared_L_0[nbytes]));
  }
  __syncthreads();
  for (i1 = threadIdx.x; i1 < opDat1SharedIndirectionSize * 1; i1 += blockDim.x) {
    opDat1SharedIndirection[i1] = 0.00000;
  }
  for (i1 = threadIdx.x; i1 < opDat3SharedIndirectionSize * 2; i1 += blockDim.x) {
    opDat3SharedIndirection[i1] = opDat3[i1 % 2 + opDat3IndirectionMap[i1 / 2] * 2];
  }
  for (i1 = threadIdx.x; i1 < opDat4SharedIndirectionSize * 1; i1 += blockDim.x) {
    opDat4SharedIndirection[i1] = opDat4[i1 % 1 + opDat4IndirectionMap[i1 / 1] * 1];
  }
  __syncthreads();
  for (i1 = threadIdx.x; i1 < numberOfActiveThreadsCeiling; i1 += blockDim.x) {
    colour2 = -1;
    if (i1 < numberOfActiveThreads) {
      for (i2 = 0; i2 < 1; ++i2) {
        opDat1Local1[i2] = 0.00000;
        opDat1Local2[i2] = 0.00000;
        opDat1Local3[i2] = 0.00000;
      }
      opDat1vec[0] = opDat1Local1;
      opDat1vec[1] = opDat1Local2;
      opDat1vec[2] = opDat1Local3;
      opDat3vec[0] = opDat3SharedIndirection + mappingArray4[i1 + sharedMemoryOffset] * 2;
      opDat3vec[1] = opDat3SharedIndirection + mappingArray5[i1 + sharedMemoryOffset] * 2;
      opDat3vec[2] = opDat3SharedIndirection + mappingArray6[i1 + sharedMemoryOffset] * 2;
      opDat4vec[0] = opDat4SharedIndirection + mappingArray7[i1 + sharedMemoryOffset] * 1;
      opDat4vec[1] = opDat4SharedIndirection + mappingArray8[i1 + sharedMemoryOffset] * 1;
      opDat4vec[2] = opDat4SharedIndirection + mappingArray9[i1 + sharedMemoryOffset] * 1;
      L_0_modified(opDat1vec,opDat2Local,opDat3vec,opDat4vec);
      colour2 = pthrcol[i1 + sharedMemoryOffset];
    }
    for (colour1 = 0; colour1 < numOfColours; ++colour1) {
      if (colour2 == colour1) {
        for (i2 = 0; i2 < 1; ++i2) {
          opDat1SharedIndirection[i2 + mappingArray1[i1 + sharedMemoryOffset] * 1] += opDat1Local1[i2];
          opDat1SharedIndirection[i2 + mappingArray2[i1 + sharedMemoryOffset] * 1] += opDat1Local2[i2];
          opDat1SharedIndirection[i2 + mappingArray3[i1 + sharedMemoryOffset] * 1] += opDat1Local3[i2];
        }
      }
      __syncthreads();
    }
  }
  for (i1 = threadIdx.x; i1 < opDat1SharedIndirectionSize * 1; i1 += blockDim.x) {
    opDat1[i1 % 1 + opDat1IndirectionMap[i1 / 1] * 1] += opDat1SharedIndirection[i1];
  }
  for (i1 = 0; i1 < 1; ++i1) {
    ReductionDouble8(&reductionArrayDevice2[i1 + blockIdx.x * 1],opDat2Local[i1],0);
  }
}

__host__ void L_0_host(const char *userSubroutine,op_set set,op_arg opDat1,op_arg opDat2,op_arg opDat3,op_arg opDat4)
{
  int blocksPerGrid;
  int threadsPerBlock;
  int dynamicSharedMemorySize;
  int i3;
  op_arg opDatArray[10];
  int indirectionDescriptorArray[10];
  op_plan *planRet;
  int blockOffset;
  int i1;
  int i2;
  int reductionBytes;
  int reductionSharedMemorySize;
  double *reductionArrayHost2;
  op_arg opDat1tmp1;
  op_arg opDat1tmp2;
  op_arg opDat3tmp1;
  op_arg opDat3tmp2;
  op_arg opDat4tmp1;
  op_arg opDat4tmp2;
  opDat1.idx = 0;
  opDatArray[0] = opDat1;
  op_duplicate_arg(opDat1,&opDat1tmp1);
  opDat1tmp1.idx = 1;
  opDatArray[1] = opDat1tmp1;
  op_duplicate_arg(opDat1tmp1,&opDat1tmp2);
  opDat1tmp2.idx = 2;
  opDatArray[2] = opDat1tmp2;
  opDatArray[3] = opDat2;
  opDat3.idx = 0;
  opDatArray[4] = opDat3;
  op_duplicate_arg(opDat3,&opDat3tmp1);
  opDat3tmp1.idx = 1;
  opDatArray[5] = opDat3tmp1;
  op_duplicate_arg(opDat3tmp1,&opDat3tmp2);
  opDat3tmp2.idx = 2;
  opDatArray[6] = opDat3tmp2;
  opDat4.idx = 0;
  opDatArray[7] = opDat4;
  op_duplicate_arg(opDat4,&opDat4tmp1);
  opDat4tmp1.idx = 1;
  opDatArray[8] = opDat4tmp1;
  op_duplicate_arg(opDat4tmp1,&opDat4tmp2);
  opDat4tmp2.idx = 2;
  opDatArray[9] = opDat4tmp2;
  indirectionDescriptorArray[1] = 0;
  indirectionDescriptorArray[2] = 0;
  indirectionDescriptorArray[0] = 0;
  indirectionDescriptorArray[3] = -1;
  indirectionDescriptorArray[5] = 1;
  indirectionDescriptorArray[6] = 1;
  indirectionDescriptorArray[4] = 1;
  indirectionDescriptorArray[8] = 2;
  indirectionDescriptorArray[9] = 2;
  indirectionDescriptorArray[7] = 2;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_L_0,10,opDatArray,3,indirectionDescriptorArray);
  blocksPerGrid = 0;
  for (i1 = 0; i1 < planRet -> ncolors; ++i1) {
    i2 = planRet -> ncolblk[i1];
    blocksPerGrid = MAX(blocksPerGrid,i2);
  }
  reductionBytes = 0;
  reductionSharedMemorySize = 0;
  reductionArrayHost2 = ((double *)opDat2.data);
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(double ) * 1);
  reductionSharedMemorySize = MAX(reductionSharedMemorySize,sizeof(double ));
  reallocReductArrays(reductionBytes);
  reductionBytes = 0;
  opDat2.data = OP_reduct_h + reductionBytes;
  opDat2.data_d = OP_reduct_d + reductionBytes;
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      ((double *)opDat2.data)[i2 + i1 * 1] = 0.00000;
    }
  }
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(double ) * 1);
  mvReductArraysToDevice(reductionBytes);
  blockOffset = 0;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    threadsPerBlock = threadsPerBlockSize_L_0;
    dynamicSharedMemorySize = MAX(planRet -> nshared,reductionSharedMemorySize * threadsPerBlock);
    L_0_kernel<<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>>(((double *)opDat1.data_d),((double *)opDat2.data_d),((double *)opDat3.data_d),((double *)opDat4.data_d),planRet -> ind_maps[0],planRet -> ind_maps[1],planRet -> ind_maps[2],planRet -> loc_maps[0],planRet -> loc_maps[1],planRet -> loc_maps[2],planRet -> loc_maps[4],planRet -> loc_maps[5],planRet -> loc_maps[6],planRet -> loc_maps[7],planRet -> loc_maps[8],planRet -> loc_maps[9],planRet -> ind_sizes,planRet -> ind_offs,planRet -> blkmap,planRet -> offset,planRet -> nelems,planRet -> nthrcol,planRet -> thrcol,blockOffset);
    cutilSafeCall(cudaThreadSynchronize());
    blockOffset += blocksPerGrid;
  }
  mvReductArraysToHost(reductionBytes);
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      reductionArrayHost2[i2] += ((double *)opDat2.data)[i2 + i1 * 1];
    }
  }
}

__device__ void a_0_modified(double *localTensor,double *dt,double *c0[2UL],int i_r_0,int i_r_1)
{
  const double CG1[3UL][6UL] = {{(0.0915762135097707), (0.0915762135097707), (0.8168475729804585), (0.4459484909159649), (0.4459484909159649), (0.1081030181680702)}, {(0.0915762135097707), (0.8168475729804585), (0.0915762135097707), (0.4459484909159649), (0.1081030181680702), (0.4459484909159649)}, {(0.8168475729804585), (0.0915762135097707), (0.0915762135097707), (0.1081030181680702), (0.4459484909159649), (0.4459484909159649)}};
  const double d_CG1[3UL][6UL][2UL] = {{{(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}, {(1.), (0.)}}, {{(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}, {(0.), (1.)}}, {{((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}, {((-1.)), ((-1.))}}};
  const double w[6UL] = {(0.0549758718276609), (0.0549758718276609), (0.0549758718276609), (0.1116907948390057), (0.1116907948390057), (0.1116907948390057)};
  double c_q0[6UL][2UL][2UL];
  for (int i_g = 0; i_g < 6; i_g++) {
    for (int i_d_0 = 0; i_d_0 < 2; i_d_0++) {
      for (int i_d_1 = 0; i_d_1 < 2; i_d_1++) {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for (int q_r_0 = 0; q_r_0 < 3; q_r_0++) {
          c_q0[i_g][i_d_0][i_d_1] += (c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1]);
        }
      }
    }
  }
  for (int i_g = 0; i_g < 6; i_g++) {
    double ST0 = 0.0;
    ST0 += ((CG1[i_r_0][i_g] * CG1[i_r_1][i_g]) * ((c_q0[i_g][0][0] * c_q0[i_g][1][1]) + (((-1) * c_q0[i_g][0][1]) * c_q0[i_g][1][0])));
    localTensor[0] += (ST0 * w[i_g]);
  }
}

__global__ void a_0_kernel(double *opMat1,double *reductionArrayDevice1,double *opDat2,int *ind_maps2,short *mappingArray1,short *mappingArray2,short *mappingArray3,int *pindSizes,int *pindOffs,int *pblkMap,int *poffset,int *pnelems,int *pnthrcol,int *pthrcol,int blockOffset)
{
  extern __device__ __shared__ char shared_a_0[];
  __device__ __shared__ int sharedMemoryOffset;
  __device__ __shared__ int numberOfActiveThreads;
  __device__ __shared__ int nbytes;
  int blockID;
  int i1;
  int i2;
  int i3;
  int i4;
  double opDat1Local[1];
  __device__ __shared__ int *opDat2IndirectionMap;
  double *opDat2vec[3];
  __device__ __shared__ int opDat2SharedIndirectionSize;
  __device__ __shared__ double *opDat2SharedIndirection;
  for (i1 = 0; i1 < 1; ++i1) {
    opDat1Local[i1] = 0.00000;
  }
  if (threadIdx.x == 0) {
    blockID = pblkMap[blockIdx.x + blockOffset];
    numberOfActiveThreads = pnelems[blockID];
    sharedMemoryOffset = poffset[blockID];
    opDat2SharedIndirectionSize = pindSizes[0 + blockID * 1];
    opDat2IndirectionMap = ind_maps2 + pindOffs[0 + blockID * 1];
    nbytes = 0;
    opDat2SharedIndirection = ((double *)(&shared_a_0[nbytes]));
  }
  __syncthreads();
  for (i1 = threadIdx.x; i1 < opDat2SharedIndirectionSize * 2; i1 += blockDim.x) {
    opDat2SharedIndirection[i1] = opDat2[i1 % 2 + opDat2IndirectionMap[i1 / 2] * 2];
  }
  __syncthreads();
  for (i1 = threadIdx.x; i1 < numberOfActiveThreads * 9; i1 += blockDim.x) {
    opMat1[i1] = ((double )0);
    i2 = i1 / 9;
    i3 = (i1 - i2 * 9) / 3;
    i4 = i1 - (i2 * 9 + i3 * 3);
    opDat2vec[0] = opDat2SharedIndirection + mappingArray1[i2 + sharedMemoryOffset] * 2;
    opDat2vec[1] = opDat2SharedIndirection + mappingArray2[i2 + sharedMemoryOffset] * 2;
    opDat2vec[2] = opDat2SharedIndirection + mappingArray3[i2 + sharedMemoryOffset] * 2;
    a_0_modified(opMat1 + i1,opDat1Local,opDat2vec,i3,i4);
  }
  for (i1 = 0; i1 < 1; ++i1) {
    ReductionDouble8(&reductionArrayDevice1[i1 + blockIdx.x * 1],opDat1Local[i1],0);
  }
}

__host__ void a_0_host(const char *userSubroutine,op_set set,op_arg opMat1,op_arg opDat1,op_arg opDat2)
{
  int blocksPerGrid;
  int threadsPerBlock;
  int dynamicSharedMemorySize;
  int i3;
  op_arg opDatArray[4];
  int indirectionDescriptorArray[4];
  op_plan *planRet;
  int blockOffset;
  int i1;
  int i2;
  int reductionBytes;
  int reductionSharedMemorySize;
  double *reductionArrayHost1;
  op_arg opDat2tmp1;
  op_arg opDat2tmp2;
  opDatArray[0] = opDat1;
  opDat2.idx = 0;
  opDatArray[1] = opDat2;
  op_duplicate_arg(opDat2,&opDat2tmp1);
  opDat2tmp1.idx = 1;
  opDatArray[2] = opDat2tmp1;
  op_duplicate_arg(opDat2tmp1,&opDat2tmp2);
  opDat2tmp2.idx = 2;
  opDatArray[3] = opDat2tmp2;
  indirectionDescriptorArray[0] = -1;
  indirectionDescriptorArray[2] = 0;
  indirectionDescriptorArray[3] = 0;
  indirectionDescriptorArray[1] = 0;
  planRet = op_plan_get(userSubroutine,set,setPartitionSize_a_0,4,opDatArray,1,indirectionDescriptorArray);
  blocksPerGrid = 0;
  for (i1 = 0; i1 < planRet -> ncolors; ++i1) {
    i2 = planRet -> ncolblk[i1];
    blocksPerGrid = MAX(blocksPerGrid,i2);
  }
  reductionBytes = 0;
  reductionSharedMemorySize = 0;
  reductionArrayHost1 = ((double *)opDat1.data);
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(double ) * 1);
  reductionSharedMemorySize = MAX(reductionSharedMemorySize,sizeof(double ));
  reallocReductArrays(reductionBytes);
  reductionBytes = 0;
  opDat1.data = OP_reduct_h + reductionBytes;
  opDat1.data_d = OP_reduct_d + reductionBytes;
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      ((double *)opDat1.data)[i2 + i1 * 1] = 0.00000;
    }
  }
  reductionBytes += ROUND_UP(blocksPerGrid * sizeof(double ) * 1);
  mvReductArraysToDevice(reductionBytes);
  if (opMat1.mat -> lma_data == NULL) {
    op_callocDevice(((void **)(&opMat1.mat -> lma_data)),9 * set -> size * sizeof(double ));
  }
  blockOffset = 0;
  for (i3 = 0; i3 < planRet -> ncolors; ++i3) {
    blocksPerGrid = planRet -> ncolblk[i3];
    threadsPerBlock = threadsPerBlockSize_a_0;
    dynamicSharedMemorySize = MAX(planRet -> nshared,reductionSharedMemorySize * threadsPerBlock);
    a_0_kernel<<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>>(((double *)(opMat1.mat -> lma_data)),((double *)opDat1.data_d),((double *)opDat2.data_d),planRet -> ind_maps[0],planRet -> loc_maps[1],planRet -> loc_maps[2],planRet -> loc_maps[3],planRet -> ind_sizes,planRet -> ind_offs,planRet -> blkmap,planRet -> offset,planRet -> nelems,planRet -> nthrcol,planRet -> thrcol,blockOffset);
    cutilSafeCall(cudaThreadSynchronize());
    blockOffset += blocksPerGrid;
  }
  op_mat_lma_to_csr(((double *)(opMat1.mat -> lma_data)),opMat1,set);
  mvReductArraysToHost(reductionBytes);
  for (i1 = 0; i1 < blocksPerGrid; ++i1) {
    for (i2 = 0; i2 < 1; ++i2) {
      reductionArrayHost1[i2] += ((double *)opDat1.data)[i2 + i1 * 1];
    }
  }
}
