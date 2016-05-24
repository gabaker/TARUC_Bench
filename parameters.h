
// C/C++ standard includes
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
//#include<math.h>

#ifndef PARAM_CLASS_INC
#define PARAM_CLASS_INC

class BenchParams {

   public:
      // Input information
      std::string inputFile;
      bool useDefaultParams;
     
      // Output file names 
      std::string runTag;
      std::string devPropFile;
      std::string topoFile;

      // All Tests (parameter file)
      bool runAllDevices;
      bool testAllMemTypes;
      bool runBurstTests;
      bool runRangeTests; 
      bool runSustainedTests;
      bool runSocketTests;
      long numStepRepeats;
      long numRangeSteps;
      long long burstBlockSize;

      // Benchmark hardware information (combination of parameters and system topology info) 
      int nDevices;
      int nSockets;

      // Overhead memory test for allocation and deallocation of Host and Device memory
      bool runMemoryOverheadTest;
      long long rangeMemOverhead[2]; //min, max range block size (in bytes)

      // Host-Host memory bandwidth test
      bool runBandwidthTestHH;
      bool runPatternsHH;
      long long rangeHostHostBW[2]; //min, max range block size (in bytes)
      
      // Host-Device PCIe baseline bandwidth test
      bool runBandwidthTestHD;
      bool runPatternsHD;
      long long rangeHostDeviceBW[2]; //min, max range block size(in bytes)

      // Peer-to-peer device memory transfer bandwidth
      bool runBandwidthTestP2P; 
      long long rangeDeviceBW[2]; //min, max range block size(in bytes)

      // Resource congestion tests
      bool runContentionTest;
      long long contBlockSize[3];
      long long numContRepeats;

      // Class initialization and setup
      void ParseParamFile(std::string FileStr);
      void SetDefault();
      void PrintParams();      
      BenchParams() {};
      ~BenchParams() {};
      
   private:
      // File parsing utility functions for value scanning
      bool GetNextBool(std::ifstream &InFile);
      long long GetNextInteger(std::ifstream &InFile);
      std::string GetNextString(std::ifstream &InFile);
      std::string GetNextLine(std::ifstream &InFile);

} ;

#endif

