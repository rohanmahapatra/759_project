USAGE
-----

Compilation:

Since all of the microbenchmarks run from a single main function, users only need to compile the entire suite once in order to use any of the microbenchmarks.  You will need to set CUDA_DIR in the Makefile in order to properly compile the code.

Running:

The usage of the microbenchmarks is as follows:

./allSyncPrims-1kernel <syncPrim> <numLdSt> <numTBs> <numCSIters>

<syncPrim> is a string that differs for each synchronization primitive to be run:
	// Barriers use hybrid local-global synchronization
	- atomicTreeBarrUniq - atomic tree barrier
	- atomicTreeBarrUniqLocalExch - atomic tree barrier with local exchange
	// global synchronization versions
	- spinSem1 - spin lock semaphore, semaphore size 1
	- spinSemEBO1 - spin lock semaphore with exponential backoff, semaphore size 1

<numLdSt> is a positive integer representing how many loads and stores each thread will perform.  For the mutexes and semaphores, these accesses are all performed in the critical section.  For the barriers, these accesses use barriers to ensure that multiple threads are not accessing the same data.

<numTBs> is a positive integer representing the number of thread blocks (TBs) to execute.  For many of the microbenchmarks (especially the barriers), this number needs to be divisible by the number of SMs on the GPU.

<numCSIters> is a positive integer representing the number of iterations of the critical section.

