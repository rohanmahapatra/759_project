#ifndef __CUDALOCKSBARRIERATOMIC_CU__
#define __CUDALOCKSBARRIERATOMIC_CU__

#include "cudaLocks.h"

inline __device__ void cudaBarrierAtomicSub(unsigned int * globalBarr,
                                            int * done,
                                            // numBarr represents the number of
                                            // TBs going to the barrier
                                            const unsigned int numBarr,
                                            int backoff,
                                            const bool isMasterThread, bool *local_sense1, bool *global_sense)
{
  __syncthreads();
  bool s = ~local_sense1[blockIdx.x];
  local_sense1[blockIdx.x] = s;
  if (isMasterThread)
  {
    //bool s = ~local_sense1[blockIdx.x];
    //local_sense1[blockIdx.x] = s;
  
    //*done = 0;
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      atomicInc effectively adds 1 to atomic for each TB that's part of the
      barrier.  For the local barrier, this requires using the per-CU
      locations.
    */
    atomicInc(globalBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  //while (global_sense != s && intermediate_sense != s)
  while (*global_sense != s)
  {
    if (isMasterThread)
    {
      /*
        Once all of the TBs on this SM have incremented the value at atomic,
        then the value (for the local barrier) should be equal to the # of TBs
        on this SM.  Once that is true, then we want to reset the atomic to 0
        and proceed because all of the TBs on this SM have reached the local
        barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        // locally
        __threadfence_block();
        //*last_block = blockIdx.x;
        *global_sense = s;
      }
      else { // increase backoff to avoid repeatedly hammering global barrier
             // (capped) exponential backoff
               backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
            }
       
    }
    __syncthreads();
   // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (!*done) {
      for (int i = 0; i < backoff; ++i) { ; }
      __syncthreads();
    }
  
  }
}

inline __device__ void cudaBarrierAtomic(unsigned int * barrierBuffers,
                                         // numBarr represents the number of
                                         // TBs going to the barrier
                                         const unsigned int numBarr,
                                         const bool isMasterThread, bool *global_sense)
{
  unsigned int * atomic1 = barrierBuffers;
  //unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1; // done2;
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();

  //bool local_sense1[numBarr];
  bool local_sense1[1025];
  for (int i = 0; i<numBarr; i++)
    local_sense1[i] = true;  


  cudaBarrierAtomicSub(atomic1, &done1, numBarr, backoff, isMasterThread, local_sense1, global_sense);
  // second barrier is necessary to provide a facesimile for a sense-reversing
  // barrier
  //cudaBarrierAtomicSub(atomic2, &done2, numBarr, backoff, isMasterThread);
}

// does local barrier amongst all of the TBs on an SM
inline __device__ void cudaBarrierAtomicSubLocal(unsigned int * perSMBarr,
                                                 int * done,
                                                 const unsigned int numTBs_thisSM,
                                                 const bool isMasterThread, bool *local_sense, int *last_block, bool * sense)
{
  __syncthreads();
  bool s = ~local_sense[blockIdx.x];
  local_sense[blockIdx.x] = s;
  if (isMasterThread)
  {
   
  
    //*done = 0;
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      atomicInc effectively adds 1 to atomic for each TB that's part of the
      barrier.  For the local barrier, this requires using the per-CU
      locations.
    */
    atomicInc(perSMBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  //while (global_sense != s && intermediate_sense != s)
  while (s != *sense)
  {
    if (isMasterThread)
    {
      /*
        Once all of the TBs on this SM have incremented the value at atomic,
        then the value (for the local barrier) should be equal to the # of TBs
        on this SM.  Once that is true, then we want to reset the atomic to 0
        and proceed because all of the TBs on this SM have reached the local
        barrier.
      */
      if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        // locally
        __threadfence_block();
        *last_block = blockIdx.x;
        *sense = s;
      }
    }
    __syncthreads();
  }
}

// does local barrier amongst all of the TBs on an SM
inline __device__ void cudaBarrierAtomicLocal(unsigned int * perSMBarrierBuffers,
                                              const unsigned int smID,
                                              const unsigned int numTBs_thisSM,
                                              const bool isMasterThread,
                                              const int MAX_BLOCKS, int *last_block, bool *sense)
{
  // each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
  // start at barrierBuffers[smID*MAX_BLOCKS]
  unsigned int * atomic1 = perSMBarrierBuffers + (smID * MAX_BLOCKS);


  //unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1; //, done2;
  //printf("Atomic 1: %d \n", atomic1);
  
  //bool local_sense[numTBs_thisSM];
  bool local_sense[1025];
  
  for (int i = 0; i<numTBs_thisSM; i++)
    local_sense[i] = true;  


  cudaBarrierAtomicSubLocal(atomic1, &done1, numTBs_thisSM, isMasterThread, local_sense, last_block, sense);
  // second barrier is necessary to approproximate a sense-reversing barrier
  //cudaBarrierAtomicSubLocal(atomic2, &done2, numTBs_thisSM, isMasterThread);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helper(unsigned int * barrierBuffers,
                                   unsigned int * perSMBarrierBuffers,
                                   const unsigned int numBlocksAtBarr,
                                   const int smID,
                                   const int perSM_blockID,
                                   const int numTBs_perSM,
                                   const bool isMasterThread,
                                   const int MAX_BLOCKS) {

  int last_block = blockIdx.x;  // by default
  bool sense = true;
  bool global_sense = true;
  if (numTBs_perSM > 1) {
    cudaBarrierAtomicLocal(perSMBarrierBuffers, smID, numTBs_perSM,isMasterThread, MAX_BLOCKS, &last_block, &sense);

    
    // only 1 TB per SM needs to do the global barrier since we synchronized
    // the TBs locally first
    if (threadIdx.x == last_block) {
      cudaBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread, &global_sense);
    }

    // all TBs on this SM do a local barrier to ensure global barrier is
    // reached
    while (sense != global_sense)
    {
      ;
    }
    //__threadfence_block;   
    //cudaBarrierAtomicLocal(perSMBarrierBuffers, smID, numTBs_perSM,
                          // isMasterThread, MAX_BLOCKS, last_block);
  } else { // if only 1 TB on the SM, no need for the local barriers
    cudaBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread, &global_sense);
    //cudaBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread);
  }
}

#endif
