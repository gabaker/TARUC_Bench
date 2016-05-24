
//cuda headers and helper functions
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

// Newer c++ timing lib 
// Does not compile with older C++ compiler versions (i.e. RHEL 6 standard g++ version)
#ifdef USING_CPP
#include <chrono>
#else
#include <sys/time.h>
#endif

#ifndef TIMER_CLASS_INC
#define TIMER_CLASS_INC
class Timer
{
   public:
      cudaStream_t stream;
      bool UseHostTimer;
      void StartTimer();
      void StopTimer();
      float ElapsedTime();
      void SetHostTiming(bool HostTimer);      

      Timer(bool UseHostTimer);
      ~Timer();

   private:
 
      #ifdef USING_CPP
      std::chrono::high_resolution_clock::time_point start_c, stop_c;
      #else
      struct timeval stop_t, start_t, total_t;
      #endif
      cudaEvent_t start_e, stop_e; 
};
#endif
