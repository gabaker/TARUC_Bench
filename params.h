
#ifndef BENCH_HEADER_INC
#define BENCH_HEADER_INC
#include "bench.h"
#endif

#ifndef PARAM_CLASS_INC
#define PARAM_CLASS_INC

class BenchParams {


   public:

   std::string resultsFile;
   std::string inputFile;
   bool useDefaultParams;

   bool printDevProps;
   std::string devPropFile;

   std::string topoFile;
   bool runTopoAware;

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

