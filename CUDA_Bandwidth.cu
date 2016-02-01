// CUDA API and includes
#include<cuda_runtime.h>

// C/C++ standard includes
#include<memory>
#include<iostream>
#include<stdio.h>
#include<string>
#include<vector>
#include<time.h>
#include<string>
#include<fstream>
#include<iostream>
#include<ios>
#include<vector>

#include<sys/time.h>
//#include<chrono>

#ifdef USING_CPP
#include<chrono>
#include<tuple>
#endif
// OpenMP threading includes
#include<omp.h>

// NUMA Locality includes
//#include<hwloc.h>
#define MILLI_TO_MICRO (1000.0)
#define NANO_TO_MICRO (1000.0)

typedef struct TestParams {
   std::string resultsFile;
   std::string inputFile;
   bool useDefaultParams;

   bool printDevProps;
   std::string devPropFile;

   int nDevices;

   // Overhead memory test for allocation and deallocation of Host and Device memory
   bool runMemoryOverheadTest;
   bool runAllDevices;
   long rangeMemOverhead[3]; //min, max and step size (in bytes)
 
   // Device-Peer PCIe Baseline bandwidth test
   bool runHostDeviceBandwidthTest;
   bool varyBlockSizeHD;
   bool usePinnedHD;
   bool runBurstHD;
   bool runSustainedHD;
   long rangeHostDeviceBW[3]; //min, max and step size (in bytes)

   // Peer-to-peer device memory transfer bandwidth
   bool runP2PBandwidthTest;
   bool varyBlockSizeP2P;
   bool runBurstP2P;
   bool runSustainedP2P;
   long rangeDeviceP2P[3]; //min, max and step size (in bytes)

   // PCIe Congestion tests
   bool runPCIeCongestionTest;

   // CUDA kernel task scalability and load balancing
   bool runTaskScalabilityTest;

} TestParams;

void RunBandwidthTestSuite(TestParams &params);

void PrintDeviceProps(cudaDeviceProp *props, TestParams &params);
void TestMemoryOverhead(cudaDeviceProp *props, TestParams &params);
void TestHostDeviceBandwidth(cudaDeviceProp *props, TestParams &params);
void TestP2PDeviceBandwidth(cudaDeviceProp *props, TestParams &params);
void TestPCIeCongestion(cudaDeviceProp *props, TestParams &params);
void TestTaskScalability(cudaDeviceProp *props, TestParams &params);
void ParseTestParameters(TestParams &params);

void SetDefaultParams(TestParams &params); 
void GetAllDeviceProps(cudaDeviceProp *props, int dCount);
void ResetDevices(int numToReset);
void SetDefaultParams(TestParams &params); 
void PrintTestParams(TestParams &params);
void getNextLine(std::ifstream &inFile, std::string &lineStr);
void printResults(std::ofstream &outFile, std::vector<long> &steps, std::vector<std::vector<float> > &results, TestParams &params); 

int main (int argc, char **argv) {
   TestParams params;
 
   
   std::cout << "\nStarting Multi-GPU Performance Test Suite...\n" << std::endl; 

   // Determine the number of recognized CUDA enabled devices
   cudaGetDeviceCount(&(params.nDevices));

   if (params.nDevices <= 0) {
      std::cout << "No devices found...aborting benchmarks." << std::endl;
      exit(-1);
   }

   // Setup benchmark parameters
   if (argc == 1) { //No input file, use default parameters
   
      SetDefaultParams(params);
   
   } else if (argc == 2) { //Parse input file
   
      params.inputFile = std::string(argv[1]);
      ParseTestParameters(params);
   
   } else { //Unknown input parameter list, abort test
      std::cout << "Aborting test: Incorrect number of input parameters" << std::endl;
      exit(-1);
   }


   PrintTestParams(params);
   RunBandwidthTestSuite(params);

   return 0;
}

void RunBandwidthTestSuite(TestParams &params) {
   cudaDeviceProp *props = (cudaDeviceProp *) calloc (sizeof(cudaDeviceProp), params.nDevices);

   // Aquire device properties for each CUDA enabled GPU
   GetAllDeviceProps(props, params.nDevices);

   if (params.runMemoryOverheadTest != false ) {
      
      TestMemoryOverhead(props, params);
   
   }

   if (params.runHostDeviceBandwidthTest != false) {

      TestHostDeviceBandwidth(props, params);

   }

   if (params.runP2PBandwidthTest != false) {  
      
      TestP2PDeviceBandwidth(props, params);
   
   }

   if (params.runPCIeCongestionTest != false) {

      TestPCIeCongestion(props, params);

   }

   if (params.runTaskScalabilityTest != false) { 

      TestTaskScalability(props, params);

   }

   // Output device properties for each CUDA enabled GPU
   if (params.printDevProps != false) {
      PrintDeviceProps(props, params);
   }

   std::cout << "\n\nBenchmarks complete!\n" << std::endl;

   free(props);
}

void TestMemoryOverhead(cudaDeviceProp *props, TestParams &params) {
      // Create CUDA runtime events used to time device operations
      cudaEvent_t start_e, stop_e; 
      cudaEventCreate(&start_e);
      cudaEventCreate(&stop_e);

      char *deviceMem = NULL;
      char *hostUnPinnedMem = NULL;
      char *hostPinnedMem = NULL;
      float eTime = 0.0;
      int nDevices = params.nDevices;

      std::vector<long> blockSteps;
//      std::vector<std::vector<float> > devData;
      std::vector<std::vector<float> > overheadData;

   // Only run overhead device cases on a single device
      // default to device 0
      if (!params.runAllDevices)
         nDevices = 1;

      // Memory overhead test will run for each device utilizing the cudaMalloc and cudaFree functions
      // on the first iteration of the look, assuming there is atleast one device, the host will run the 
      // pinned and un-pinned memory tests
      long stepNum = 0;
      long stepSize = (params.rangeMemOverhead[0] / 10 > 0) ? params.rangeMemOverhead[0] : 1; 
      for ( long chunkSize = params.rangeMemOverhead[0]; 
            chunkSize <= params.rangeMemOverhead[1]; 
            chunkSize += stepSize) {
 
         blockSteps.push_back(chunkSize); 
         std::vector<float> chunkData;
//         std::vector<float> chunkDeviceData;

         for (int currDev = 0; currDev < nDevices; currDev++) {
            cudaSetDevice(currDev);

            // CASE 1 & 2: Host memory overhead
            // Host test only runs the first time
            if (currDev == 0) {

               // Pinned
               //allocation
               cudaEventRecord(start_e);                             //record start call
               cudaMallocHost((void **) &hostPinnedMem, chunkSize);  //malloc pinned memory
               cudaEventRecord(stop_e);                              //record finish call
               cudaEventSynchronize(stop_e);                         //sync all cuda calls before finish event
               cudaEventElapsedTime(&eTime, start_e, stop_e);        //calculate function call time
               chunkData.push_back((float) eTime/* * MILLI_TO_MICRO*/);
               //deallocation
               cudaEventRecord(start_e);
               cudaFreeHost((void *) hostPinnedMem);
               cudaEventRecord(stop_e);
               cudaEventSynchronize(stop_e);
               cudaEventElapsedTime(&eTime, start_e, stop_e);
               chunkData.push_back((float) eTime /* * MILLI_TO_MICRO*/);


               // Unpinned
               #ifdef USING_CPP
               auto start_t = std::chrono::high_resolution_clock::now();
               hostUnPinnedMem = (char *) malloc(chunkSize);
               auto stop_t = std::chrono::high_resolution_clock::now();
               auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_t - start_t);

               chunkData.push_back((float) duration.count() * NANO_TO_MICRO);

               start_t = std::chrono::high_resolution_clock::now();
               free(hostUnPinnedMem);
               stop_t = std::chrono::high_resolution_clock::now(); 
               duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_t - start_t);
               chunkData.push_back((float) duration.count() * NANO_TO_MICRO);
               
               #else
               struct timeval stop_t, start_t, total_t;
               
               //allocation
               gettimeofday(&start_t, NULL);
               hostUnPinnedMem = (char *) malloc(chunkSize); 
               gettimeofday(&stop_t, NULL);
               timersub(&stop_t, &start_t, &total_t);
               chunkData.push_back(((float) total_t.tv_usec));
               //deallocation
               gettimeofday(&start_t, NULL);
               free(hostUnPinnedMem); 
               gettimeofday(&stop_t, NULL); 
               timersub(&stop_t, &start_t, &total_t); 
               chunkData.push_back(((float) total_t.tv_usec));
               #endif

            }

            // CASE 3: Allocation of device memory
            cudaEventRecord(start_e);
            cudaMalloc((void **) &deviceMem, chunkSize); 
            cudaEventRecord(stop_e);

            cudaEventSynchronize(stop_e);
            cudaEventElapsedTime(&eTime, start_e, stop_e);
            chunkData.push_back((float)eTime  /*MILLI_TO_MICRO*/);

            // CASE 4: DeAllocation of device memory
            cudaEventRecord(start_e);
            cudaFree(deviceMem); 
            cudaEventRecord(stop_e);

            cudaEventSynchronize(stop_e);   
            cudaEventElapsedTime(&eTime, start_e, stop_e); 
            chunkData.push_back((float) eTime /* MILLI_TO_MICRO*/);

         }
         //devData.push_back(chunkDeviceData);
         overheadData.push_back(chunkData);
         //chunkDeviceData.clear();
         chunkData.clear();
        
         //stepNum++; 
         //do 10 steps then check next range
         if (stepNum && (stepNum % (params.rangeMemOverhead[2] - 1)) == 0)
            stepSize *= 10;
         stepNum++;
      }

      // cleanup CUDA runtime events
      cudaEventDestroy(start_e);
      cudaEventDestroy(stop_e);

      //Print test Data
/*      std::string fileName = param.resultsFile + "_overhead.csv";
      std::ofstream devResultsFile(fileName.c_str()); 
      printResults(devResultsFile, blockSteps, devData, params);
*/
      std::string dataFileName = params.resultsFile + "_overhead.csv";
      std::ofstream overheadResultsFile(dataFileName.c_str());
      printResults(overheadResultsFile, blockSteps, overheadData, params);
}

void printResults(std::ofstream &outFile, std::vector<long> &steps, std::vector<std::vector<float> > &results, TestParams &params) {
   //std::cout.setf(std::ios::showpoint);
   
   if (!outFile.is_open()) {
      std::cout << "Failed to open file to print results" << std::endl;
      return;
   }
   std::vector<std::vector<float> >::iterator iter_o;
   std::vector<float>::iterator iter_i;
   std::vector<long>::iterator iter_l = steps.begin();
   
   for (iter_o = results.begin(); iter_o != results.end(); ++iter_o) {
      outFile << std::fixed << *iter_l++ << ",";
      for (iter_i = (*iter_o).begin(); iter_i != (*iter_o).end(); ++iter_i) {
         outFile << std::fixed << *iter_i;
         if (iter_i + 1 != (*iter_o).end())
            outFile << ",";
      }
      outFile << std::endl;
   }
}

void TestHostDeviceBandwidth(cudaDeviceProp *props, TestParams &params) {
   std::cout << "Running host-device bandwidth test" << std::endl;
   //printf("\nRunning bandwidth test for %s on bus %d\n", props[0].name, props[0].pciBusID);
}

void TestP2PDeviceBandwidth(cudaDeviceProp *props, TestParams &params){
   std::cout << "Running P2P device bandwidth test" << std::endl;
}

void TestPCIeCongestion(cudaDeviceProp *props, TestParams &params) {
   std::cout << "Running PCIe congestion test" << std::endl;
}


void TestTaskScalability(cudaDeviceProp *props, TestParams &params) {
   std::cout << "Running task scalability test" << std::endl;
}

void getNextLine(std::ifstream &inFile, std::string &lineStr) {
   // get lines of the input file untill the first character of the line is not a dash
   // dashes represent comments
   do { 
      if (inFile) 
         std::getline(inFile, lineStr);
   } while (inFile && lineStr[0] == '-');
}

bool getNextLineBool(std::ifstream &inFile, std::string &lineStr) {
   do { 
      if (inFile) 
         std::getline(inFile, lineStr);
   } while (inFile && lineStr[0] == '-');

   return ((lineStr.find("alse") >= lineStr.length()) ? true : false); 
}

// Function for parsing user provided input file. 
// Users must adhere to input file structure provided 
// in the sample input file to insure correct parsing
void ParseTestParameters(TestParams &params) {
   std::string lineStr;
   std::ifstream inFile(params.inputFile.c_str());

   params.useDefaultParams = false;

   getNextLine(inFile, lineStr); //resultsFile
   params.resultsFile = lineStr.substr(lineStr.find ('=') + 1);

   params.printDevProps = getNextLineBool(inFile, lineStr); //printDeviceProps
   getNextLine(inFile, lineStr);
   params.devPropFile = lineStr.substr(lineStr.find ('=') + 1); //devPropFile
  
   params.runMemoryOverheadTest = getNextLineBool(inFile, lineStr); //runMemoryOverheadTest
   params.runAllDevices = getNextLineBool(inFile, lineStr); //runAllDevices 
   for (int i = 0; i < 3; i++) {
      getNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      params.rangeMemOverhead[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }

   params.runHostDeviceBandwidthTest = getNextLineBool(inFile, lineStr); //runHostDeviceBandwidthTest
   params.varyBlockSizeHD = getNextLineBool(inFile, lineStr); //varyBlockSizeHD
   params.usePinnedHD = getNextLineBool(inFile, lineStr); //usePinnedHD
   params.runBurstHD = getNextLineBool(inFile, lineStr); //runBurstHD
   params.runSustainedHD = getNextLineBool(inFile, lineStr); //runSustainedHD
   for (int i = 0; i < 3; i++) {
      getNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      params.rangeHostDeviceBW[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }

   params.runP2PBandwidthTest = getNextLineBool(inFile, lineStr); //runP2PBandwidthTest
   params.varyBlockSizeP2P = getNextLineBool(inFile, lineStr); //varyBlockSizeP2P
   params.runBurstP2P = getNextLineBool(inFile, lineStr); //runBurstHD
   params.runSustainedP2P = getNextLineBool(inFile, lineStr); //runSustainedHD
   for (int i = 0; i < 3; i++) {
      getNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      params.rangeDeviceP2P[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }
   
   params.runPCIeCongestionTest = getNextLineBool(inFile, lineStr); //runPCIeCongestionTest
   params.runTaskScalabilityTest = getNextLineBool(inFile, lineStr); //runTaskScalabilityTest
   
}

//TODO:hacky print function; fix this
void PrintTestParams(TestParams &params) {

   std::string paramFileName = "benchmark_params.out";
   std::ofstream outParamFile(paramFileName.c_str());

   outParamFile << std::boolalpha;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "---------------------- Test Parameters ---------------------" << std::endl; 
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Input File:\t\t\t" << params.inputFile << std::endl;
   outParamFile << "Output file:\t\t\t" << params.resultsFile << std::endl;
   outParamFile << "Using Defaults:\t\t\t" << params.useDefaultParams << std::endl;  
   outParamFile << "Printing Device Props:\t\t" << params.printDevProps << std::endl;
   outParamFile << "Device Property File:\t\t" << params.devPropFile << std::endl;
   outParamFile << "Device Count:\t\t\t" << params.nDevices << std::endl;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Run Memory Overhead Test:\t" << params.runMemoryOverheadTest << std::endl;
   outParamFile << "Use all Devices:\t\t" << params.runAllDevices << std::endl;
   outParamFile << "Allocation Range: \t\t";
   outParamFile << params.rangeMemOverhead[0] << "," << params.rangeMemOverhead[1];
   outParamFile << "," << params.rangeMemOverhead[2] << " (min,max,step)" << std::endl;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Run Host-Device Bandwidth Test:\t" << params.runHostDeviceBandwidthTest << std::endl;
   outParamFile << "Vary Block Size:\t\t" << params.varyBlockSizeHD << std::endl;
   outParamFile << "Use Pinned Host Mem:\t\t" << params.usePinnedHD << std::endl;
   outParamFile << "Burst Mode:\t\t\t" << params.runBurstHD << std::endl;
   outParamFile << "Sustained Mode:\t\t\t" << params.runSustainedHD << std::endl;
   outParamFile << "Allocation Range:\t\t"; 
   outParamFile << params.rangeHostDeviceBW[0] << "," << params.rangeHostDeviceBW[1] << ","; 
   outParamFile << params.rangeHostDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Run P2P Bandwidth Test:\t\t" << params.runP2PBandwidthTest << std::endl;
   outParamFile << "Vary Block Size:\t\t" << params.varyBlockSizeP2P << std::endl;
   outParamFile << "Burst Mode:\t\t\t" << params.runBurstP2P << std::endl;
   outParamFile << "Sustained Mode:\t\t\t" << params.runSustainedP2P << std::endl;
   outParamFile << "Allocation Range:\t\t";
   outParamFile << params.rangeDeviceP2P[0] << "," << params.rangeDeviceP2P[1] << ",";
   outParamFile << params.rangeDeviceP2P[2] << " (min,max,step)" << std::endl;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Run PCIe CongestionTest:\t" << params.runPCIeCongestionTest << std::endl;
   outParamFile << "------------------------------------------------------------" << std::endl; 
   outParamFile << "Run Task Scalability Test:\t" << params.runTaskScalabilityTest << std::endl; 
   outParamFile << "------------------------------------------------------------" << std::endl;    
   outParamFile << std::noboolalpha;

   //read params out to command line
   outParamFile.close();
   std::string contents;
   std::ifstream inFile(paramFileName.c_str());
   while (std::getline(inFile,contents)) {
      std::cout << contents << std::endl;
   } 
   inFile.close();

}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void SetDefaultParams(TestParams &params) {

   params.resultsFile = "results";
   params.inputFile = "none";
   params.useDefaultParams = true;

   params.printDevProps = true;
   params.devPropFile = "device_info.out";

   params.runMemoryOverheadTest = true; 
   params.runAllDevices = true;
   params.rangeMemOverhead[0] = 1;
   params.rangeMemOverhead[1] = 1000001;
   params.rangeMemOverhead[2] = 10000;
   
   params.runHostDeviceBandwidthTest = false;
   params.varyBlockSizeHD = true;
   params.usePinnedHD = true;
   params.runBurstHD  = true;
   params.runSustainedHD = true;
   params.rangeHostDeviceBW[0] = 1;
   params.rangeHostDeviceBW[1] = 1024;
   params.rangeHostDeviceBW[2] = 2; 
  
   params.runP2PBandwidthTest = false;
   params.varyBlockSizeP2P = true;
   params.runBurstP2P = true;
   params.runSustainedP2P = true;
   params.rangeDeviceP2P[0] = 1;
   params.rangeDeviceP2P[1] = 2024;
   params.rangeDeviceP2P[2] = 2;
   
   params.runPCIeCongestionTest = false;
   
   params.runTaskScalabilityTest = false;
}

// Prints the device properties out to file based named depending on the 
void PrintDeviceProps(cudaDeviceProp *props, TestParams &params) {
   std::cout << "\nSee " << params.devPropFile << " for information about your device's properties." << std::endl; 

   std::ofstream deviceProps(params.devPropFile.c_str());

   deviceProps << "Device Properties:" << std::endl;

   deviceProps.close();
}

// Creates an array of cudaDeviceProp structs with populated data
// located in a pre-allocated section of memory
void GetAllDeviceProps(cudaDeviceProp *props, int dCount) {
   for (int i = 0; i < dCount; ++i) {
      cudaGetDeviceProperties(&props[i], i);
   }
}

// function for cleaning up device state including profile data
// to be used before and after any test in benchmark suite.
void ResetDevices(int numToReset) {
   for (int devNum = 0; devNum < numToReset; ++devNum) {
      cudaSetDevice(devNum);
      cudaDeviceReset();
   }
}

