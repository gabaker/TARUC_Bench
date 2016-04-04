
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
   devPropFile = resultsFile + "_device_info.out";
   topoFile = resultsFile + "_topology.out";
   printDevProps = GetNextLineBool(inFile, lineStr); //printDeviceProps
   
   // All Tests
   runAllDevices = GetNextLineBool(inFile, lineStr); //runAllDevices 
   usePinnedMem = GetNextLineBool(inFile, lineStr); //usePinnedMem 
   runBurstTests = GetNextLineBool(inFile, lineStr); //runBurstTests 
   runRangeTests = GetNextLineBool(inFile, lineStr); //runRangeTests
   runSustainedTests = GetNextLineBool(inFile, lineStr); //runSustainedTests
   runSocketTests = GetNextLineBool(inFile, lineStr); //runSocketTests
   GetNextLine(inFile, lineStr); //numStepRepeats
   int eqIdx = lineStr.find("=") + 1;
   numStepRepeats = std::atol(lineStr.substr(eqIdx).c_str());
   GetNextLine(inFile, lineStr); //burstBlockSize
   eqIdx = lineStr.find("=") + 1;
   burstBlockSize = pow(2, 20) * std::atol(lineStr.substr(eqIdx).c_str());
 
   // Memory Overhead Test
   runMemoryOverheadTest = GetNextLineBool(inFile, lineStr); //runMemoryOverheadTest
   for (int i = 0; i < 3; i++) { //rangeMemOverhead
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeMemOverhead[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }

   // Host-Host Bandwidth Test
   runHHBandwidthTest = GetNextLineBool(inFile, lineStr); //runHHBandwidthTest
   runPatternsHH = GetNextLineBool(inFile, lineStr); //runPatternsHH
   for (int i = 0; i < 3; i++) { //rangeHostHostBW
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeHostHostBW[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }

   // Host-Device Bandwidth Test
   runHDBandwidthTest = GetNextLineBool(inFile, lineStr); //runHDBandwidthTest
   runPatternsHD = GetNextLineBool(inFile, lineStr); //runPatternsHD
   for (int i = 0; i < 3; i++) { //rangeHostDeviceBW
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeHostDeviceBW[i] = std::atoll(lineStr.substr(eqIdx).c_str());
   }

   // P2P Bandwidth Test
   runP2PBandwidthTest = GetNextLineBool(inFile, lineStr); //runP2PBandwidthTest
   for (int i = 0; i < 3; i++) { //rangeDeviceP2P
      GetNextLine(inFile, lineStr);
      eqIdx = lineStr.find("=") + 1;
      rangeDeviceBW[i] = std::atoll(lineStr.substr(eqIdx).c_str());
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
   devPropFile = resultsFile + "_device_info.out";
   topoFile = resultsFile + "_topology.out";

   runAllDevices = true;
   usePinnedMem = true;
   runBurstTests = true;
   runRangeTests = true; 
   runSustainedTests = true;
   runSocketTests = true;
   numStepRepeats = 5;
   burstBlockSize = pow(2,26);
   runMemoryOverheadTest = true; 
   
   rangeMemOverhead[0] = 10;
   rangeMemOverhead[1] = pow(2,26);
   rangeMemOverhead[2] = 10;
   runPatternsHH = false;
   
   runHDBandwidthTest = true;
   rangeHostDeviceBW[0] = 10;
   rangeHostDeviceBW[1] = pow(2,26);
   rangeHostDeviceBW[2] = 10; 
   runPatternsHH = false;

   runP2PBandwidthTest = true;
   rangeDeviceBW[0] = 10;
   rangeDeviceBW[1] = pow(2,26);
   rangeDeviceBW[2] = 10;
   
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
   outParamStr << "Burst Tests:\t\t\t" << runBurstTests << std::endl;
   outParamStr << "Burst Block Size:\t\t" << burstBlockSize << std::endl;
   outParamStr << "Ranged Tests:\t\t\t" << runRangeTests << std::endl;
   outParamStr << "Sustained Tests:\t\t" << runSustainedTests << std::endl;
   outParamStr << "Test All Sockets:\t\t" << runSocketTests << std::endl;
   outParamStr << "Number Repeated Steps:\t" << numStepRepeats << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Memory Overhead Test --------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runMemoryOverheadTest << std::endl;
   outParamStr << "Allocation Range: \t\t";
   outParamStr << rangeMemOverhead[0] << "," << rangeMemOverhead[1];
   outParamStr << "," << rangeMemOverhead[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Host Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runHHBandwidthTest << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHH << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostHostBW[0] << "," << rangeHostHostBW[1] << ","; 
   outParamStr << rangeHostHostBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Device Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runHDBandwidthTest << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHD << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostDeviceBW[0] << "," << rangeHostDeviceBW[1] << ","; 
   outParamStr << rangeHostDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- P2P Bandwidth Test ----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runP2PBandwidthTest << std::endl;
   outParamStr << "Allocation Range:\t\t";
   outParamStr << rangeDeviceBW[0] << "," << rangeDeviceBW[1] << ",";
   outParamStr << rangeDeviceBW[2] << " (min,max,step)" << std::endl;
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
   std::string paramFileName = "./results/" + resultsFile + "_parameters.out";
   std::ofstream outParamFile(paramFileName.c_str());
   outParamFile << outParamStr.str() << std::endl;

   //read params out to command line
   outParamFile.close();
}



