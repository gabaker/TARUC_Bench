
//#ifndef PARAMS_CLASS_INC
//#define PARAMS_CLASS_INC
#include "parameters.h"
//#endif

long long BenchParams::GetNextInteger(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);
   
   lineStr = lineStr.substr(lineStr.find ('=') + 1);
   lineStr = lineStr.substr(lineStr.find_first_not_of(' '));
   
   return std::atoll(lineStr.c_str());
}

bool BenchParams::GetNextBool(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);

   return ((lineStr.find("alse") >= lineStr.length()) ? true : false); 
}

std::string BenchParams::GetNextString(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);

   lineStr = lineStr.substr(lineStr.find('=') + 1);

   return lineStr.substr(lineStr.find_first_not_of(' '));
}

std::string BenchParams::GetNextLine(std::ifstream &inFile) {
   // get lines of the input file until the first character of the line is not a dash
   // dashes represent comments
   std::string lineStr;
 
   do { 
      if (inFile) 
         std::getline(inFile, lineStr);
   } while (inFile && lineStr[0] == '-');

   return lineStr;
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
   runTag  = GetNextString(inFile); //runTag
   devPropFile = runTag + "_device_info.out";
   topoFile = runTag + "_topology.out";
   printDevProps = GetNextBool(inFile); //printDeviceProps
   
   // All Tests
   runAllDevices = GetNextBool(inFile); //runAllDevices 
   testAllMemTypes = GetNextBool(inFile); //testAllMemTypes 
   runBurstTests = GetNextBool(inFile); //runBurstTests 
   runRangeTests = GetNextBool(inFile); //runRangeTests
   runSustainedTests = GetNextBool(inFile); //runSustainedTests
   runSocketTests = GetNextBool(inFile); //runSocketTests
   numStepRepeats = (long) GetNextInteger(inFile); //numStepRepeats
   burstBlockSize = GetNextInteger(inFile); //burstBlockSize
 
   // Memory Overhead Test
   runMemoryOverheadTest = GetNextBool(inFile); //runMemoryOverheadTest
   for (int i = 0; i < 3; i++) //rangeMemOverhead
      rangeMemOverhead[i] = GetNextInteger(inFile); 

   // Host-Host Bandwidth Test
   runBandwidthTestHH = GetNextBool(inFile); //runHHBandwidthTest
   runPatternsHH = GetNextBool(inFile); //runPatternsHH
   for (int i = 0; i < 3; i++) //rangeHostHostBW
      rangeHostHostBW[i] = GetNextInteger(inFile); 

   // Host-Device Bandwidth Test
   runBandwidthTestHD = GetNextBool(inFile); //runHDBandwidthTest
   runPatternsHD = GetNextBool(inFile); //runPatternsHD
   for (int i = 0; i < 3; i++)  //rangeHostDeviceBW
      rangeHostDeviceBW[i] = GetNextInteger(inFile); 

   // P2P Bandwidth Test
   runBandwidthTestP2P = GetNextBool(inFile); //runP2PBandwidthTest
   for (int i = 0; i < 3; i++) //rangeDeviceP2P
      rangeDeviceBW[i] = GetNextInteger(inFile);
   
   // Memory System Congestion Test 
   runCongestionTest = GetNextBool(inFile); //runCongestionTest
   
   // Task Scalability
   runUsageTest = GetNextBool(inFile); //runMemUsageTest
  
}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void BenchParams::SetDefault() {
   
   inputFile = "none";
   useDefaultParams = true;

   runTag = "results";
   printDevProps = true;
   devPropFile = runTag + "_device_info.out";
   topoFile = runTag + "_topology.out";

   runAllDevices = true;
   testAllMemTypes = true;
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

   runBandwidthTestHH = true;
   rangeHostDeviceBW[0] = 10;
   rangeHostDeviceBW[1] = pow(2,26);
   rangeHostDeviceBW[2] = 10; 
   runPatternsHH = false;
   
   runBandwidthTestHD = true;
   rangeHostDeviceBW[0] = 10;
   rangeHostDeviceBW[1] = pow(2,26);
   rangeHostDeviceBW[2] = 10; 
   runPatternsHD = false;

   runBandwidthTestP2P = true;
   rangeDeviceBW[0] = 10;
   rangeDeviceBW[1] = pow(2,26);
   rangeDeviceBW[2] = 10;
   
   runCongestionTest = false;
   
   runUsageTest = false;
}

void BenchParams::PrintParams() {
   std::stringstream outParamStr;
   outParamStr << std::boolalpha;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "------------------------- Test Parameters -----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Input File:\t\t\t" << inputFile << std::endl;
   outParamStr << "Using Defaults:\t\t\t" << useDefaultParams << std::endl;  
   outParamStr << "Run Tag:\t\t\t" << runTag << std::endl;
   outParamStr << "Printing Device Props:\t\t" << printDevProps << std::endl;
   outParamStr << "Device Property File:\t\t" << devPropFile << std::endl;
   outParamStr << "Topology File:\t\t\t" << topoFile << std::endl;  
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "---------------------------- All Tests --------------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Use all Devices:\t\t" << runAllDevices << std::endl;
   outParamStr << "Device Count:\t\t\t" << nDevices << std::endl;
   outParamStr << "Test All Host Mem Types:\t\t" << testAllMemTypes << std::endl;
   outParamStr << "Burst Tests:\t\t\t" << runBurstTests << std::endl;
   outParamStr << "Burst Block Size:\t\t" << burstBlockSize << std::endl;
   outParamStr << "Ranged Tests:\t\t\t" << runRangeTests << std::endl;
   outParamStr << "Sustained Tests:\t\t" << runSustainedTests << std::endl;
   outParamStr << "Test All Sockets:\t\t" << runSocketTests << std::endl;
   outParamStr << "Number Repeated Steps:\t\t" << numStepRepeats << std::endl;
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
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestHH << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHH << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostHostBW[0] << "," << rangeHostHostBW[1] << ","; 
   outParamStr << rangeHostHostBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Device Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestHD << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHD << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostDeviceBW[0] << "," << rangeHostDeviceBW[1] << ","; 
   outParamStr << rangeHostDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- P2P Bandwidth Test ----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestP2P << std::endl;
   outParamStr << "Allocation Range:\t\t";
   outParamStr << rangeDeviceBW[0] << "," << rangeDeviceBW[1] << ",";
   outParamStr << rangeDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Pipeline Congestion ---------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Congestion Test:\t\t" << runCongestionTest << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------------- Memory Usage  ------------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Memory Usage Test:\t\t" << runUsageTest << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << std::noboolalpha;

   //Print benchmark parameters to string
   std::cout << "\n" << outParamStr.str();

   // Print benchmark parameters to output file
   std::string paramFileName = "./results/" + runTag + "_parameters.out";
   std::ofstream outParamFile(paramFileName.c_str());
   outParamFile << outParamStr.str() << std::endl;

   //read params out to command line
   outParamFile.close();
}



