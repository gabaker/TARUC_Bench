# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <omp.h>
# include <iostream>
# include "topology.h"

#define STREAM_ARRAY_SIZE 80000000

#define NTIMES	10

#define OFFSET	0

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

static double  avgtime[4] = {0}, 
               maxtime[4] = {0},
               mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {2,2,3,3};

/*
static double	bytes[4] = {
      2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
      2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
      3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
      3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE};
*/
double mysecond();
void checkSTREAMresults(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c);

int main() {
   int quantum, checktick();
   int BytesPerWord;
   double t, times[4][NTIMES];

   /* --- SETUP --- determine precision and check timing --- */
   printf(HLINE);
   printf("STREAM version $Revision: 5.10 $\n");
   printf(HLINE);
   BytesPerWord = sizeof(STREAM_TYPE);
   printf("This system uses %d bytes per array element.\n", BytesPerWord);
   printf(HLINE);
   printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
   printf("Memory per array = %.1f MiB (= %.1f GiB).\n", 
   BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
   printf("Total memory required = %.1f MiB (= %.1f GiB).\n", 
                                    (3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
                                    (3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
   printf("Each kernel will be executed %d times.\n", NTIMES);
   printf(" The *best* time for each kernel (excluding the first iteration)\n"); 
   printf(" will be used to compute the reported bandwidth.\n");
   printf(HLINE);

   SystemTopo topo;

   
   /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
   STREAM_TYPE scalar = 3.0;
   int NumHostMemTypes = 2;
   int NumTestCases = 7;
   int MaxThreadCount = omp_get_max_threads();
   int MaxTestedThreadCount = topo.NumPUsPerSocket() * topo.NumSockets();

   std::cout << "\t";
   printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
   long long BlockSize = STREAM_ARRAY_SIZE * sizeof(double);
   long long NumSteps =  STREAM_ARRAY_SIZE;
   for (int memType = 0; memType < NumHostMemTypes; ++memType) {
      //std::cout << "Mem Type: " << memType << std::endl;
      #ifdef SHARED_BLOCK
      static STREAM_TYPE * __restrict__ a = (STREAM_TYPE *) malloc(BlockSize);
      static STREAM_TYPE * __restrict__ b = (STREAM_TYPE *) malloc(BlockSize);
      static STREAM_TYPE * __restrict__ c = (STREAM_TYPE *) malloc(BlockSize);
      #endif

      for (int testCase = 0; testCase < NumTestCases; ++testCase) {
         std::cout << "Mem Types: " << memType << " Case: " << testCase << std::endl;
         for (int numThreads = 1; numThreads <= MaxTestedThreadCount; numThreads *= 2) {

            #ifdef SHARED_BLOCK
            omp_set_num_threads(MaxThreadCount);  

            #pragma omp parallel for 
            for (ssize_t j=0; j<STREAM_ARRAY_SIZE; j++) {
               a[j] = 1.0;
               b[j] = 2.0;
               c[j] = 0.0;
            }

            #pragma omp parallel for
            for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++)
               a[j] = 2.0E0 * a[j];
            #endif

            omp_set_num_threads(numThreads);  
         
            for (int k=0; k<NTIMES; k++) { 
               double startTime = 0; 
               #ifndef SHARED_BLOCK
               NumSteps = (long long) STREAM_ARRAY_SIZE / numThreads;
               BlockSize = (long long) NumSteps * sizeof(double);

               #pragma omp parallel 
               {
                  int threadIdx = omp_get_thread_num();
                  STREAM_TYPE * __restrict__ a;               
                  STREAM_TYPE * __restrict__ b;               
                  STREAM_TYPE * __restrict__ c;     
          
                  int aNode, bNode, cNode;
                  int socket, core;

                  // CASE 0: Memory and execution on different sockets, all use same socket 
                  // CASE 1: Memory and execution on differnet sockets, threads alternate socket used
                  // CASE 2: Memory and execution on same socket, all threads use same socket 
                  // CASE 3: Memory and execution on same socket, threads use different sockets
                  // CASE 4: Execution on a single socket; memory on both sockets
                  // CASE 5: Cross node copy, same socket execution
                  // CASE 6: Cross node copy, both socket execution
                  if (testCase == 0) {
                     core = threadIdx % topo.NumCoresPerSocket();
                     socket = 0;
                     aNode = 1;
                     bNode = 1;
                     cNode = 1;
                  } else if (testCase == 1) {
                     core = threadIdx / topo.NumSockets();
                     socket = threadIdx % topo.NumSockets();
                     aNode = (threadIdx + 1) % topo.NumSockets();
                     bNode = (threadIdx + 1) % topo.NumSockets();
                     cNode = (threadIdx + 1) % topo.NumSockets();
                  } else if (testCase == 2) {
                     core = threadIdx % topo.NumCoresPerSocket();
                     socket = 0;
                     aNode = 0;
                     bNode = 0;
                     cNode = 0;
                  } else if (testCase == 3) {
                     core = threadIdx / topo.NumSockets();
                     socket = threadIdx % topo.NumSockets();
                     aNode = threadIdx % topo.NumSockets();
                     bNode = threadIdx % topo.NumSockets();
                     cNode = threadIdx % topo.NumSockets();
                  } else if (testCase == 4) {
                     core = threadIdx % topo.NumCoresPerSocket();
                     socket = 0;
                     aNode = threadIdx % topo.NumSockets();
                     bNode = threadIdx % topo.NumSockets();
                     cNode = threadIdx % topo.NumSockets();
                  } else if (testCase == 5) {
                     core = threadIdx % topo.NumCoresPerSocket();
                     socket = 0;
                     aNode = threadIdx % topo.NumSockets();
                     bNode = threadIdx % topo.NumSockets();
                     cNode = (threadIdx + 1) % topo.NumSockets();
                  } else if (testCase == 6) {
                     core = threadIdx / topo.NumSockets();
                     socket = threadIdx % topo.NumSockets();
                     aNode = threadIdx % topo.NumSockets();
                     bNode = threadIdx % topo.NumSockets();
                     cNode = (threadIdx + 1) % topo.NumSockets();
                  } else {
                     std::cout << "Error: unrecognized test case: " << testCase << std::endl; 
                     exit(-1);
                  }     
                 
                  topo.PinCoreBySocket(socket, core); 
                  
                  if (memType == 0) {
                     a = (STREAM_TYPE *) topo.AllocMemByNode(aNode, BlockSize);
                     b = (STREAM_TYPE *) topo.AllocMemByNode(bNode, BlockSize);
                     c = (STREAM_TYPE *) topo.AllocMemByNode(cNode, BlockSize);
                  } else { //memType == 1
                     a = (STREAM_TYPE *) topo.AllocPinMemByNode(aNode, BlockSize);
                     b = (STREAM_TYPE *) topo.AllocPinMemByNode(bNode, BlockSize);
                     c = (STREAM_TYPE *) topo.AllocPinMemByNode(cNode, BlockSize);
                  }
                  if (testCase != 7) {
                     topo.SetHostMem(a, 1.0 * 2.0E0, BlockSize);
                     topo.SetHostMem(b, 2.0, BlockSize);
                     topo.SetHostMem(c, 0.0, BlockSize);
                  } 
                  
                  topo.PinNode(socket);                  

                  // COPY ----------------------------------------------------
                  #pragma omp barrier 
                  #pragma omp single                  
                  #endif
                  {
                     times[0][k] = 0.0;
                     startTime = mysecond();
                  }

                  #ifdef SHARED_BLOCK 
                  #pragma omp parallel for
                  #endif
                  for (ssize_t j=0; j < NumSteps; j++)
                     c[j] = a[j];

                  #ifndef SHARED_BLOCK
                  #pragma omp barrier 
                  #pragma omp critical
                  #endif
                  {
                     times[0][k] += mysecond() - startTime;
                  }

                  // SCALE -------------------------------------------------------
                  #ifndef SHARED_BLOCK                 
                  #pragma omp barrier 
                  #pragma omp single                  
                  #endif 
                  {
                     times[1][k] = 0.0;
                     startTime = mysecond();
                  }

                  #ifdef SHARED_BLOCK 
                  #pragma omp parallel for
                  #endif
                  for (ssize_t j=0; j < NumSteps; j++)
                     c[j] = scalar*a[j];

                  #ifndef SHARED_BLOCK
                  #pragma omp barrier 
                  #pragma omp critical
                  #endif
                  {
                     times[1][k] += mysecond() - startTime;
                  }

                  // ADD -------------------------------------------------------
                  #ifndef SHARED_BLOCK                 
                  #pragma omp barrier 
                  #pragma omp single                  
                  #endif 
                  {
                     times[2][k] = 0.0;
                     startTime = mysecond();
                  }

                  #ifdef SHARED_BLOCK 
                  #pragma omp parallel for
                  #endif
                  for (ssize_t j=0; j < NumSteps; j++)
                     c[j] = a[j]+b[j];

                  #ifndef SHARED_BLOCK

                  #pragma omp barrier 
                 
                  #pragma omp critical
                  #endif
                  {
                     times[2][k] += mysecond() - startTime;
                  }

                  // TRIAD -----------------------------------------------------
                  #ifndef SHARED_BLOCK                 
                  #pragma omp barrier 
                  #pragma omp single                  
                  #endif 
                  {
                     times[3][k] = 0.0;
                     startTime = mysecond();
                  }

                  #ifdef SHARED_BLOCK 
                  #pragma omp parallel for
                  #endif
                  for (ssize_t j=0; j < NumSteps; j++)
                     c[j] = a[j]+scalar*b[j];

                  #ifndef SHARED_BLOCK

                  #pragma omp barrier 
                 
                  #pragma omp critical
                  #endif
                  {
                     times[3][k] += mysecond() - startTime;
                  }
                  
                  #ifndef SHARED_BLOCK 
                  if (memType == 0) {
                     topo.FreeHostMem(a, BlockSize);
                     topo.FreeHostMem(b, BlockSize);
                     topo.FreeHostMem(c, BlockSize);
                  } else { // memType == 1
                     topo.FreePinMem(a, BlockSize);
                     topo.FreePinMem(b, BlockSize);
                     topo.FreePinMem(c, BlockSize);
                  }
               }
               #endif
            }
            
            /*	--- SUMMARY --- */
            for (int j=0; j<4; j++) {
               avgtime[j] = 0;
               maxtime[j] = 0;
               mintime[j] = FLT_MAX;
            }

            /* note -- skip first iteration */
            for (int k=1; k<NTIMES; k++) {
               for (int j=0; j<4; j++) {
                  avgtime[j] = avgtime[j] + times[j][k];
                  mintime[j] = MIN(mintime[j], times[j][k]);
                  maxtime[j] = MAX(maxtime[j], times[j][k]);
               }
            }
             
            for (int j=0; j<4; j++) {
               avgtime[j] = avgtime[j]/(double)(NTIMES-1);
               std::cout << numThreads << "\t";
               #ifndef SHARED_BLOCK
               printf("%s%12.4f  %11.6f  %11.6f  %11.6f\n", label[j],
                                                            (1.0E-09 * bytes[j] * BlockSize * (double) numThreads) / mintime[j] * (double) numThreads,
                                                            avgtime[j] / (double) numThreads,
                                                            mintime[j] / (double) numThreads,
                                                            maxtime[j] / (double) numThreads);
               #else
               printf("%s%12.4f  %11.6f  %11.6f  %11.6f\n", label[j],
                                                            (1.0E-09 * bytes[j] * BlockSize) / mintime[j],
                                                            avgtime[j] / (double) numThreads,
                                                            mintime[j] / (double) numThreads,
                                                            maxtime[j] / (double) numThreads);
               #endif
            }
            printf(HLINE);
         }
      }
      
      #ifdef SHARED_BLOCK
      free(a);
      free(b);
      free(c);
      #endif
   }


   /* --- Check Results --- */
   //checkSTREAMresults(a,b,c);
   //printf(HLINE);

   return 0;
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

# define	M	20

int checktick() {
   int i, minDelta, Delta;
   double t1, t2, timesfound[M];

   /*  Collect a sequence of M unique time values from the system. */

   for (i = 0; i < M; i++) {
	   t1 = mysecond();
	   while( ((t2=mysecond()) - t1) < 1.0E-6 );
	      timesfound[i] = t1 = t2;
   }

   /*
    * Determine the minimum difference between these M values.
    * This result will be our estimate (in microseconds) for the
    * clock granularity.
    */

   minDelta = 1000000;
   for (i = 1; i < M; i++) {
	   Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	   minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
}



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond() {
   struct timeval tp;
   struct timezone tzp;
   int i;

   i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
   STREAM_TYPE aj,bj,cj,scalar;
   STREAM_TYPE aSumErr,bSumErr,cSumErr;
   STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
   double epsilon;
   ssize_t j;
   int k, ierr, err;

   /* reproduce initialization */
   aj = 1.0;
   bj = 2.0;
   cj = 0.0;
   /* a[] is modified during timing check */
   aj = 2.0E0 * aj;
   /* now execute timing loop */
   scalar = 3.0;
   for (k=0; k<NTIMES; k++) {
      cj = aj;
      bj = scalar*cj;
      cj = aj+bj;
      aj = bj+scalar*cj;
   }

   /* accumulate deltas between observed and expected results */
   aSumErr = 0.0;
   bSumErr = 0.0;
   cSumErr = 0.0;
   for (j=0; j<STREAM_ARRAY_SIZE; j++) {
      aSumErr += abs(a[j] - aj);
      bSumErr += abs(b[j] - bj);
      cSumErr += abs(c[j] - cj);
      // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
   }
   
   aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
   bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
   cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

   if (sizeof(STREAM_TYPE) == 4) {
      epsilon = 1.e-6;
   } else if (sizeof(STREAM_TYPE) == 8) {
      epsilon = 1.e-13;
   } else {
      printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
      epsilon = 1.e-6;
   }

   err = 0;
   if (abs(aAvgErr/aj) > epsilon) {
      err++;
      printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
      ierr = 0;
      for (j=0; j<STREAM_ARRAY_SIZE; j++) {
         if (abs(a[j]/aj-1.0) > epsilon) {
            ierr++;
   #ifdef VERBOSE
            if (ierr < 10) {
               printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                  j,aj,a[j],abs((aj-a[j])/aAvgErr));
            }
   #endif
         }
      }
      printf("     For array a[], %d errors were found.\n",ierr);
   }
   if (abs(bAvgErr/bj) > epsilon) {
      err++;
      printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
      printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
      ierr = 0;
      for (j=0; j<STREAM_ARRAY_SIZE; j++) {
         if (abs(b[j]/bj-1.0) > epsilon) {
            ierr++;
   #ifdef VERBOSE
            if (ierr < 10) {
               printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                  j,bj,b[j],abs((bj-b[j])/bAvgErr));
            }
   #endif
         }
      }
      printf("     For array b[], %d errors were found.\n",ierr);
   }
   if (abs(cAvgErr/cj) > epsilon) {
      err++;
      printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
      printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
      ierr = 0;
      for (j=0; j<STREAM_ARRAY_SIZE; j++) {
         if (abs(c[j]/cj-1.0) > epsilon) {
            ierr++;
   #ifdef VERBOSE
            if (ierr < 10) {
               printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                  j,cj,c[j],abs((cj-c[j])/cAvgErr));
            }
   #endif
         }
      }
      printf("     For array c[], %d errors were found.\n",ierr);
   }
   if (err == 0) {
      printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
   }
   #ifdef VERBOSE
   printf ("Results Validation Verbose Results: \n");
   printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
   printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
   printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
   #endif
}


