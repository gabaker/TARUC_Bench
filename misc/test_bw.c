#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
//#include "topology.h"
#include<sys/mman.h>

#define N 100000000
#define L 100000000

#ifdef INNER
//static double *a, *b, *c;
static double a[N], b[N], c[N];
#endif
int main(void) {
   double bw; 
   double totalTime = 0.0;
   register int numRepeats = 100;
   omp_set_num_threads(32);
   
   #ifdef INNER
   register long long k;

   //a = (double *) malloc(sizeof(double) * N);
   //b = (double *) malloc(sizeof(double) * N);
   //c = (double *) malloc(sizeof(double) * N);
   for (register long long j = 0; j < N; ++j) {
      a[j] = 2.0E0;
      b[j] = 2.0E0;
      c[j] = 1.0E0;
   }

   mlock(a, N*sizeof(double));
   mlock(b, N*sizeof(double));
   mlock(c, N*sizeof(double));
   
   for (k = 0; k < numRepeats; ++k) {
      struct timezone tz;
      struct timeval stop_t, start_t, total_t;
      
      gettimeofday(&start_t, &tz);


      #pragma omp parallel for
      for (register long long j = 0; j < N; ++j)
         c[j] = a[j];
      

      gettimeofday(&stop_t, &tz);
      timersub(&stop_t, &start_t, &total_t); 
      double time = (float) total_t.tv_usec + (float) total_t.tv_sec * 1.0e6;
      
      if (k > 0)
         totalTime += time * 1.0E-6;
   }
   
   totalTime /= (double) (numRepeats - 1);

   bw = 1.0E-3 * 2.0 * ((double) (N * sizeof(double) * 1.0E-06)) / (double)(totalTime);
   #else

   {
      static double *a, *b, *c;
      a = (double *) malloc(sizeof(double) * L);
      b = (double *) malloc(sizeof(double) * L);
      c = (double *) malloc(sizeof(double) * L);
      double time = 0.0;
      mlock(a, L*sizeof(double));
      mlock(b, L*sizeof(double));
      mlock(c, L*sizeof(double));
      
      struct timezone tz;
      struct timeval stop_t, start_t, total_t;

      for (register int i = 0; i < numRepeats; ++i) {

         #pragma omp parallel for
         for (register long long j = 0; j < L; ++j) {
            a[j] = 2.0E0;
            b[j] = 2.0E0;
            c[j] = 1.0E0;
         }
         register double * a_ptr = a;
         register double * b_ptr = b;
         register double * c_ptr = c;

         gettimeofday(&start_t, &tz);

         #pragma omp parallel for
         /*for (register long long j = 0; j < L; ++j)
            *(a_ptr + j) = 1.0;
         */
         for (register double *j = a; j < (a + N); ++j)
            *j = 1.0;

         gettimeofday(&stop_t, &tz);
         timersub(&stop_t, &start_t, &total_t); 
         time += (float) total_t.tv_usec * 1.0E-6 + (float) total_t.tv_sec;
      }
      //#pragma omp barrier

      //#pragma omp atomic
      totalTime += time / (numRepeats);
   }

   bw = 1.0E-3 * 2.0 * ((double) (L * sizeof(double) * 1.0E-06)) / (double)(totalTime);

   #endif

   printf("%lf\n", totalTime);
   printf("%lf\n", bw);

   return 0;
}
