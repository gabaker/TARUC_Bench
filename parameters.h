
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "benchmark.h"
#endif

#ifndef PARAM_CLASS_INC
#define PARAM_CLASS_INC

#include<math.h>
class BenchParams {

   public:
      std::string inputFile;
      bool useDefaultParams;
      
      std::string resultsFile;
      bool printDevProps;
      std::string devPropFile;
      std::string topoFile;

      bool runAllDevices;
      bool usePinnedMem;

      int nDevices;

      // Overhead memory test for allocation and deallocation of Host and Device memory
      bool runMemoryOverheadTest;
      long numStepRepeatsOH;
      long long rangeMemOverhead[3]; //min, max and step size (in bytes)
    
      // Host-Device PCIe Baseline bandwidth test
      bool runHDBandwidthTest;
      bool runRangeTestHD; 
      bool runBurstHD;
      bool runSustainedHD;
      bool runAllPatternsHD;
      long numCopiesPerStepHD;
      long long rangeHostDeviceBW[3]; //min, max and step size (in bytes)

      // Peer-to-peer device memory transfer bandwidth
      bool runP2PBandwidthTest; 
      bool runRangeTestP2P; 
      bool runBurstP2P;
      bool runSustainedP2P;
      long numCopiesPerStepP2P;
      long long rangeDeviceP2P[3]; //min, max and step size (in bytes)

      // PCIe Congestion tests
      bool runPCIeCongestionTest;

      // CUDA kernel task scalability and load balancing
      bool runTaskScalabilityTest;

      void ParseParamFile(std::string fileStr);
      void SetDefault();
      void PrintParams();      
      BenchParams() {};
      ~BenchParams() {};
      
   private:
      void GetNextLine(std::ifstream &inFile, std::string &lineStr);
      bool GetNextLineBool(std::ifstream &inFile, std::string &lineStr);

} ;

#endif

