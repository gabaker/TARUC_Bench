//CUDA API and includes
#include<cuda_runtime.h>

// C/C++ standard includes
#include<memory>
#include<iostream>
#include<stdio.h>
#include<string>

//OpenMP threading includes
#include<omp.h>

// NUMA Locality includes
//#include<hwloc.h>

typedef struct TestParams {
   std::string resultsFileName;
   std::string inputFile;

   bool printDevProps;
   std::string devicePropFileName;  

   //Overhead memory test for allocation and deallocation of Host and Device memory
   bool runMemoryOverheadTest;
   long rangeMemOverhead[3]; //min, max and step size (in bytes)
 
   //Device-Peer PCIe Baseline bandwidth test
   bool runHostDeviceBandwidthTest;
   bool varyBlockSizeHD;
   bool usePinnedHD;
   bool runBurstHD;
   bool runSustainedHD;
   long rangeHostDeviceBWTest[3]; //min, max and step size (in bytes)

   //Peer-to-peer device memory transfer bandwidth
   bool runP2PBandwidthTest;
   bool varyBlockSizeP2P;
   bool runBurstP2P;
   bool runSustainedP2P;
   long rangeHostDeviceP2P[3]; //min, max and step size (in bytes)

   //PCIe Congestion tests
   bool runPCIeCongestionTest;

   //CUDA kernel task scalability and load balancing
   bool runTaskScalabilityTest;

} TestParams;

void RunBandwidthTestSuite(int argc, char **argv);

void PrintDeviceProps(cudaDeviceProp *props, int dCount, TestParams &params);
void TestMemoryOverhead(cudaDeviceProp *props, int dCount, TestParams &params);
void TestHostDeviceBandwidth(cudaDeviceProp *props, int dCount, TestParams &params);
void TestP2PDeviceBandwidth(cudaDeviceProp *props, int dCount, TestParams &params);
void TestPCIeCongestion(cudaDeviceProp *props, int dCount, TestParams &params);
void TestTaskScalability(cudaDeviceProp *props, int dCount, TestParams &params);

void SetDefaultParams(TestParams &params); 
void GetAllDeviceProps(cudaDeviceProp *props, int dCount);
void ResetDevices(int numToReset);
void SetDefaultParams(TestParams &params); 

int main (int argc, char **argv) {

   int nDevices = 0;

   cudaGetDeviceCount(&nDevices);

   if (nDevices <= 0) {
   
      printf("No devices Found\n");
      return 0;
   
   } else {

      RunBandwidthTestSuite(argc, argv);
   }

   return 0;
}

void RunBandwidthTestSuite(int argc, char **argv) {
   int nDevices = 0;
   TestParams params;

   // If command line parameters provide an input file skip this and do input parsing
   if (1) {
      SetDefaultParams(params);
   } else {
      //TODO: Parse input parameters
      
   }

   //Determine the number of recognized CUDA enabled devices
   cudaGetDeviceCount(&nDevices);
   cudaDeviceProp *props = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), nDevices);

   //Aquire device properties for each CUDA enabled GPU
   GetAllDeviceProps(props, nDevices);

   //Output device properties for each CUDA enabled GPU
   if (params.printDevProps != false) {
      PrintDeviceProps(props, nDevices, params);
   }

   if (params.runMemoryOverheadTest != false ) {
      
      TestMemoryOverhead(props, nDevices, params);
   
/*      cudaEvent_t start, stop; 
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      ResetDevices(nDevices);

      char *blockPtr = NULL;
      float eTime = 0;
      
      for (int chunkSize = 2; chunkSize <= 1024; chunkSize += 2) {
         cudaEventRecord(start);
         cudaMalloc((void **) &blockPtr, chunkSize);
         cudaEventRecord(stop);

         cudaEventSynchronize(stop);
         
         cudaEventElapsedTime(&eTime, start, stop);

         printf("%f\n", eTime);
   
         cudaFree(blockPtr);      

      }

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

*/


   }

   if (params.runHostDeviceBandwidthTest != false) {

      TestHostDeviceBandwidth(props, nDevices, params);

   }

   if (params.runP2PBandwidthTest != false) {  
      
      TestP2PDeviceBandwidth(props, nDevices, params);
   
   }

   if (params.runPCIeCongestionTest != false) {

      TestPCIeCongestion(props, nDevices, params);

   }

   if (params.runTaskScalabilityTest != false) { 

      TestTaskScalability(props, nDevices, params);

   }

}

void TestMemoryOverhead(cudaDeviceProp *props, int dCount, TestParams &params) {
      cudaEvent_t start, stop; 
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      ResetDevices(dCount);

      char *blockPtr = NULL;
      float eTime = 0.0;
 
      //for (int currDev = 0; currDev < dCount; currDev++) {
         //printf("Running device %d (ID) of %d (total)\n", 0, dCount);
         //cudaSetDevice(1);

     
         //for ( long chunkSize = params.rangeMemOverhead[0]; 
         //      chunkSize <= params.rangeMemOverhead[1]; 
         //      chunkSize += params.rangeMemOverhead[2]) {
         for (int chunkSize = 0; chunkSize <= 100000; chunkSize += 25000) {
            cudaEventRecord(start);
            cudaMalloc((void **) &blockPtr, chunkSize);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
         
            cudaEventElapsedTime(&eTime, start, stop);

            printf("%lf\n", eTime);
   
            cudaFree(blockPtr); 

            cudaDeviceSynchronize();

         }

      //}

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, int dCount, TestParams &params) {

   //printf("\nRunning bandwidth test for %s on bus %d\n", props[0].name, props[0].pciBusID);

}

void TestP2PDeviceBandwidth(cudaDeviceProp *props, int dCount, TestParams &params){


}

void TestPCIeCongestion(cudaDeviceProp *props, int dCount, TestParams &params) {


}


void TestTaskScalability(cudaDeviceProp *props, int dCount, TestParams &params) {


}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void SetDefaultParams(TestParams &params) {

   params.resultsFileName = "Results.csv";
   params.inputFile = "Input.txt";

   params.printDevProps = false;
   params.devicePropFileName = "DeviceInfo.txt";

   params.runMemoryOverheadTest = true; 
   params.rangeMemOverhead[0] = 1;
   params.rangeMemOverhead[1] = 65535;
   params.rangeMemOverhead[2] = 1024;
   
   params.runHostDeviceBandwidthTest = false;
   params.varyBlockSizeHD = true;
   params.usePinnedHD = true;
   params.runBurstHD  = true;
   params.runSustainedHD = true;
   params.rangeHostDeviceBWTest[0] = 1;
   params.rangeHostDeviceBWTest[1] = 1024;
   params.rangeHostDeviceBWTest[2] = 2; 
  
   params.runP2PBandwidthTest = false;
   params.varyBlockSizeP2P = true;
   params.runBurstP2P = true;
   params.runSustainedP2P = true;
   params.rangeHostDeviceP2P[0] = 1;
   params.rangeHostDeviceP2P[1] = 2024;
   params.rangeHostDeviceP2P[2] = 2;
   
   params.runPCIeCongestionTest = false;
   
   params.runTaskScalabilityTest = false;
}

//Prints the device properties out to file based named depending on the 
void PrintDeviceProps(cudaDeviceProp *props, int dCount, TestParams &params) {
   printf("See %s for information about your device's properties\n", params.devicePropFileName.c_str());

}

// Creates an array of cudaDeviceProp structs with populated data
// located in a pre-allocated section of memory
void GetAllDeviceProps(cudaDeviceProp *props, int dCount) {
   for (int i = 0; i < dCount; ++i) {
      cudaGetDeviceProperties(&props[i], i);
   }
}

//function for cleaning up device state including profile data
//to be used before and after any test in benchmark suite.
void ResetDevices(int numToReset) {
   for (int devNum = 0; devNum < numToReset; ++devNum) {
      cudaSetDevice(devNum);
      cudaDeviceReset();
   }
}

