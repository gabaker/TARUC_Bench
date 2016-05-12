
//cuda headers and helper functions
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"


#ifdef USING_CPP
#include <chrono>
#endif

#include <sys/time.h>
#include <time.h>

#ifndef TIMER_CLASS_INC
#define TIMER_CLASS_INC
class Timer
{
   public:
      cudaStream_t stream;
      
      void StartTimer();
      void StopTimer();
      float ElapsedTime();
      void ResetTimer();
      
      Timer(bool UseHostTimer);
      ~Timer();

   private:
   
      #ifdef USING_CPP
      std::chrono::high_resolution_clock::time_point start_c, stop_c;
      #else
      struct timeval stop_t, start_t, total_t;
      #endif
      cudaEvent_t start_e, stop_e; 
      const bool UseHostTimer;
};
#endif
