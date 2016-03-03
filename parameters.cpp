
//#ifndef PARAMS_CLASS_INC
//#define PARAMS_CLASS_INC
#include "parameters.h"
//#endif

bool BenchParams::GetNextLineBool(std::ifstream &inFile, std::string &lineStr) {
  do { 
      if (inFile) 
         std::getline(inFile, lineStr);
   } while (inFile && lineStr[0] == '-');

   return ((lineStr.find("alse") >= lineStr.length()) ? true : false); 
}


void BenchParams::GetNextLine(std::ifstream &inFile, std::string &lineStr) {
   // get lines of the input file until the first character of the line is not a dash
   // dashes represent comments
   do { 
      if (inFile) 
         std::getline(inFile, lineStr);
   } while (inFile && lineStr[0] == '-');
}

// Function for parsing user provided input file. 
// Users must adhere to input file structure provided 
// in the sample input file to insure correct parsing
void BenchParams::ParseParamFile(std::string fileStr) {
   inputFile = fileStr;

   std::string lineStr;
   std::ifstream inFile(inputFile.c_str());

   if (inFile.fail()) {
      SetDefault();      
      return;
   }

   useDefaultParams = false;

   // Benchmark Parameters 
   GetNextLine(inFile, lineStr); //resultsFile
   resultsFile = lineStr.substr(lineStr.find ('=') + 1);
   printDevProps = GetNextLineBool(inFile, lineStr); //printDeviceProps
   GetNextLine(inFile, lineStr);
   devPropFile = lineStr.substr(lineStr.find ('=') + 1); //devPropFile
   GetNextLine(inFile, lineStr);
   topoFile = lineStr.substr(lineStr.find ('=') + 1); //topoFile
   
   // All Tests
   runAllDevices = GetNextLineBool(inFile, lineStr); //runAllDevices 
   usePinnedMem = GetNextLineBool(inFile, lineStr); //usePinnedMem 
  
   // Memory Overhead Test
   runMemoryOverheadTest = GetNextLineBool(inFile, lineStr); //runMemoryOverheadTest
   GetNextLine(inFile, lineStr); //numCopiesPerStepOH
   int eqIdx = lineStr.find("=") + 1;
   numStepRepeatsOH = std::atoll(lineStr.substr(eqIdx).c_str());
   for (int i = 0; i < 3; i++) { //rangeMemOverhead
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeMemOverhead[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }

   // Host-Device Bandwidth Test
   runHDBandwidthTest = GetNextLineBool(inFile, lineStr); //runHDBandwidthTest
   runRangeTestHD = GetNextLineBool(inFile, lineStr); //runRangeTestHD
   runBurstHD = GetNextLineBool(inFile, lineStr); //runBurstHD
   runSustainedHD = GetNextLineBool(inFile, lineStr); //runSustainedHD
   runAllPatternsHD = GetNextLineBool(inFile, lineStr); //runAllPatternsHD
   GetNextLine(inFile, lineStr); //numCopiesPerStepHD
   eqIdx = lineStr.find("=") + 1;
   numCopiesPerStepHD = std::atoll(lineStr.substr(eqIdx).c_str());
   for (int i = 0; i < 3; i++) { //rangeHostDeviceBW
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeHostDeviceBW[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }

   // P2P Bandwidth Test
   runP2PBandwidthTest = GetNextLineBool(inFile, lineStr); //runP2PBandwidthTest
   runRangeTestP2P = GetNextLineBool(inFile, lineStr); //runRangeTestP2P
   runBurstP2P = GetNextLineBool(inFile, lineStr); //runBurstP2P
   runSustainedP2P = GetNextLineBool(inFile, lineStr); //runSustainedP2P
   GetNextLine(inFile, lineStr); //numCopiesPerStepP2P
   eqIdx = lineStr.find("=") + 1;
   numCopiesPerStepP2P = std::atoll(lineStr.substr(eqIdx).c_str());
   for (int i = 0; i < 3; i++) { //rangeDeviceP2P
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeDeviceP2P[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }
   
   // PCIe Congestion Test 
   runPCIeCongestionTest = GetNextLineBool(inFile, lineStr); //runPCIeCongestionTest
   
   // Task Scalability
   runTaskScalabilityTest = GetNextLineBool(inFile, lineStr); //runTaskScalabilityTest
   
}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void BenchParams::SetDefault() {
   
   inputFile = "none";
   useDefaultParams = true;

   resultsFile = "results";
   printDevProps = true;
   devPropFile = "device_info.out";
   topoFile = "none";

   runAllDevices = true;
   usePinnedMem = true;
   
   runMemoryOverheadTest = true; 
   numStepRepeatsOH = 10;
   rangeMemOverhead[0] = 10;
   rangeMemOverhead[1] = 1500000000;
   rangeMemOverhead[2] = 10;
   
   runHDBandwidthTest = false;
   runRangeTestHD = true;
   runBurstHD  = true;
   runSustainedHD = true;
   runAllPatternsHD = false;
   numCopiesPerStepHD = 10;
   rangeHostDeviceBW[0] = 10;
   rangeHostDeviceBW[1] = 1500000000;
   rangeHostDeviceBW[2] = 10; 
  
   runP2PBandwidthTest = false;
   runRangeTestP2P = true;
   runBurstP2P = true;
   runSustainedP2P = true;
   numCopiesPerStepP2P = 10;
   rangeDeviceP2P[0] = 10;
   rangeDeviceP2P[1] = 1500000000;
   rangeDeviceP2P[2] = 10;
   
   runPCIeCongestionTest = false;
   
   runTaskScalabilityTest = false;
}

void BenchParams::PrintParams() {
   std::stringstream outParamStr;
   outParamStr << std::boolalpha;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "------------------------- Test Parameters -----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Input File:\t\t\t" << inputFile << std::endl;
   outParamStr << "Using Defaults:\t\t\t" << useDefaultParams << std::endl;  
   outParamStr << "Results file:\t\t\t" << resultsFile << std::endl;
   outParamStr << "Printing Device Props:\t\t" << printDevProps << std::endl;
   outParamStr << "Device Property File:\t\t" << devPropFile << std::endl;
   outParamStr << "Topology File:\t\t\t" << topoFile << std::endl;  
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "---------------------------- All Tests --------------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Use all Devices:\t\t" << runAllDevices << std::endl;
   outParamStr << "Device Count:\t\t\t" << nDevices << std::endl;
   outParamStr << "Use Pinned Host Mem:\t\t" << usePinnedMem << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Memory Overhead Test --------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Memory Overhead Test:\t" << runMemoryOverheadTest << std::endl;
   outParamStr << "Number of Copies Per Step:\t" << numStepRepeatsOH << std::endl;
   outParamStr << "Allocation Range: \t\t";
   outParamStr << rangeMemOverhead[0] << "," << rangeMemOverhead[1];
   outParamStr << "," << rangeMemOverhead[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Device Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Host-Device Bandwidth Test:\t" << runHDBandwidthTest << std::endl;
   outParamStr << "Ranged Memory Sizes:\t\t" << runRangeTestHD << std::endl;
   outParamStr << "Burst Mode:\t\t\t" << runBurstHD << std::endl;
   outParamStr << "Sustained Mode:\t\t\t" << runSustainedHD << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runAllPatternsHD << std::endl;
   outParamStr << "Number of Copies Per Step:\t" << numCopiesPerStepHD << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostDeviceBW[0] << "," << rangeHostDeviceBW[1] << ","; 
   outParamStr << rangeHostDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- P2P Bandwidth Test ----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run P2P Bandwidth Test:\t\t" << runP2PBandwidthTest << std::endl;
   outParamStr << "Ranged Memory Size:\t\t" << runRangeTestP2P << std::endl;
   outParamStr << "Burst Mode:\t\t\t" << runBurstP2P << std::endl;
   outParamStr << "Sustained Mode:\t\t\t" << runSustainedP2P << std::endl;
   outParamStr << "Number of Copies Per Step:\t" << numCopiesPerStepP2P << std::endl;
   outParamStr << "Allocation Range:\t\t";
   outParamStr << rangeDeviceP2P[0] << "," << rangeDeviceP2P[1] << ",";
   outParamStr << rangeDeviceP2P[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Pipeline Congestion ---------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run PCIe CongestionTest:\t" << runPCIeCongestionTest << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "------------------------- Task Scalability ----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Task Scalability Test:\t" << runTaskScalabilityTest << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << std::noboolalpha;

   //Print benchmark parameters to string
   std::cout << "\n" << outParamStr.str();

   // Print benchmark parameters to output file
   std::string paramFileName = "./results/bench_params.out";
   std::ofstream outParamFile(paramFileName.c_str());
   outParamFile << outParamStr.str() << std::endl;

   //read params out to command line
   outParamFile.close();
}
