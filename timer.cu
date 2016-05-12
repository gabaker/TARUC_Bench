#include "timer.h"            

void Timer::StartTimer() {

   if (UseHostTimer) {
      #ifdef USING_CPP
      start_c = std::chrono::high_resolution_clock::now();
      #else
      gettimeofday(&start_t, NULL);
      #endif
   } else {
      checkCudaErrors(cudaEventRecord(start_e, stream));
   }
}

void Timer::StopTimer() {
   if (UseHostTimer) {
      #ifdef USING_CPP
      stop_c = std::chrono::high_resolution_clock::now(); 
      #else
      gettimeofday(&stop_t, NULL);
      #endif 
   } else {
      checkCudaErrors(cudaEventRecord(stop_e, stream));   
   }
}

// Returns elasped time in microseconds
float Timer::ElapsedTime() {
   float time = 0.0;

   if (UseHostTimer) {
      #ifdef USING_CPP
      auto total_c = std::chrono::duration_cast<std::chrono::microseconds>(stop_c - start_c);
      time = (float) total_c.count(); 
      #else
      timersub(&stop_t, &start_t, &total_t); 
      time = (float) total_t.tv_usec + (float) total_t.tv_sec * 1.0e6;
      #endif 
   } else {      
      checkCudaErrors(cudaEventSynchronize(stop_e)); 
      checkCudaErrors(cudaEventElapsedTime(&time, start_e, stop_e)); 
      time *= 1.0e3;  
   }

   return time;
}

//TODO check if cuda events are reusable; if so delete member function
void Timer::ResetTimer() {
   if (!UseHostTimer) {
      checkCudaErrors(cudaStreamDestroy(stream));
      checkCudaErrors(cudaEventDestroy(start_e));
      checkCudaErrors(cudaEventDestroy(stop_e));

      checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));             
      checkCudaErrors(cudaEventCreate(&start_e));
      checkCudaErrors(cudaEventCreate(&stop_e));        
   } 
}

Timer::~Timer() {
   if (!UseHostTimer) {
      checkCudaErrors(cudaStreamDestroy(stream));
      checkCudaErrors(cudaEventDestroy(start_e));
      checkCudaErrors(cudaEventDestroy(stop_e));
   } 
}

Timer::Timer(bool UseHostTimer = true) : UseHostTimer(UseHostTimer) {
   if (!UseHostTimer) {
      checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));             
      checkCudaErrors(cudaEventCreate(&start_e));
      checkCudaErrors(cudaEventCreate(&stop_e)); 
   } 
}




