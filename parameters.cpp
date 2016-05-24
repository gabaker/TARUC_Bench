
#ifndef PARAMS_CLASS_INC
#define PARAMS_CLASS_INC
#include "parameters.h"
#endif

// Get integer from next line formated as follows : "Label information = 5000"
long long BenchParams::GetNextInteger(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);
   
   lineStr = lineStr.substr(lineStr.find ('=') + 1);
   lineStr = lineStr.substr(lineStr.find_first_not_of(' '));
   
   return std::atoll(lineStr.c_str());
}

// Get boolean from next line formated as follows : "Label information = true"
bool BenchParams::GetNextBool(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);

   return ((lineStr.find("alse") >= lineStr.length()) ? true : false); 
}

// Get string from next line formated as follows : "Label information = STRING_NAME_HERE"
std::string BenchParams::GetNextString(std::ifstream &inFile) {
   std::string lineStr = GetNextLine(inFile);

   lineStr = lineStr.substr(lineStr.find('=') + 1);

   return lineStr.substr(lineStr.find_first_not_of(' '));
}

// Gets next line from file that is not a comment
// Comment lines start with a dash ("-")
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
   runTag  = GetNextString(inFile);                // runTag
   devPropFile = runTag + "_device_info.out";
   topoFile = runTag + "_topology.out";
   
   // All Tests
   runAllDevices = GetNextBool(inFile);            // runAllDevices 
   testAllMemTypes = GetNextBool(inFile);          // testAllMemTypes 
   runBurstTests = GetNextBool(inFile);            // runBurstTests 
   runRangeTests = GetNextBool(inFile);            // runRangeTests
   runSustainedTests = GetNextBool(inFile);        // runSustainedTests
   runSocketTests = GetNextBool(inFile);           // runSocketTests
   numStepRepeats = (long) GetNextInteger(inFile); // numStepRepeats
   numRangeSteps = (long) GetNextInteger(inFile);  // numRangeSteps
   burstBlockSize = GetNextInteger(inFile);        // burstBlockSize
 
   // Memory Overhead Test
   runMemoryOverheadTest = GetNextBool(inFile);    // runMemoryOverheadTest
   for (int i = 0; i < 2; i++)                     // rangeMemOverhead
      rangeMemOverhead[i] = GetNextInteger(inFile); 

   // Host-Host Bandwidth Test
   runBandwidthTestHH = GetNextBool(inFile);    // runHHBandwidthTest
   runPatternsHH = GetNextBool(inFile);         // runPatternsHH
   for (int i = 0; i < 2; i++)                  // rangeHostHostBW
      rangeHostHostBW[i] = GetNextInteger(inFile); 

   // Host-Device Bandwidth Test
   runBandwidthTestHD = GetNextBool(inFile);    // runHDBandwidthTest
   runPatternsHD = GetNextBool(inFile);         // runPatternsHD
   for (int i = 0; i < 2; i++)                  // rangeHostDeviceBW
      rangeHostDeviceBW[i] = GetNextInteger(inFile); 

   // P2P Bandwidth Test
   runBandwidthTestP2P = GetNextBool(inFile);   // runP2PBandwidthTest
   for (int i = 0; i < 2; i++)                  // rangeDeviceP2P
      rangeDeviceBW[i] = GetNextInteger(inFile);
   
   // Memory System Contention Test 
   runContentionTest = GetNextBool(inFile);     // runContentionTest
   numContRepeats = GetNextInteger(inFile);     // numContRepeats
   for (int i = 0; i < 3; i++)                  // rangeCont
      contBlockSize[i] = GetNextInteger(inFile);
}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void BenchParams::SetDefault() {
   
   inputFile = "none";
   useDefaultParams = true;

   runTag = "results";
   devPropFile = runTag + "_device_info.out";
   topoFile = runTag + "_topology.out";

   runAllDevices = true;
   testAllMemTypes = true;
   runBurstTests = true;
   runRangeTests = true; 
   runSustainedTests = true;
   runSocketTests = true;
   numStepRepeats = 10;
   numRangeSteps = 10;
   burstBlockSize = 100000000;
   runMemoryOverheadTest = true; 
   
   rangeMemOverhead[0] = 1000;
   rangeMemOverhead[1] = 100000000;

   runBandwidthTestHH = true;
   runPatternsHH = true;
   rangeHostDeviceBW[0] = 1000;
   rangeHostDeviceBW[1] = 1000000000;
   
   runBandwidthTestHD = true;
   runPatternsHD = true;
   rangeHostDeviceBW[0] = 1000;
   rangeHostDeviceBW[1] = 1000000000;

   runBandwidthTestP2P = true;
   rangeDeviceBW[0] = 1000;             // 100B min block size
   rangeDeviceBW[1] = 1000000000;      // 1500MB max block size 
   
   runContentionTest = false;
   numContRepeats = 50;
   contBlockSize[0] = 100000000;    // Local Host Memory Contention Test
   contBlockSize[1] = 100000000;    // QPI Intersocket Memory Contententon Test 
   contBlockSize[2] = 100000000;    // Host-Device Memory Contention Test (PCIe) 
   
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
   outParamStr << "Device Property File:\t\t" << devPropFile << std::endl;
   outParamStr << "Topology File:\t\t\t" << topoFile << std::endl;  
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "---------------------------- All Tests --------------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Use all Devices:\t\t" << runAllDevices << std::endl;
   outParamStr << "Device Count:\t\t\t" << nDevices << std::endl;
   outParamStr << "Test All Host Mem Types:\t" << testAllMemTypes << std::endl;
   outParamStr << "Burst Tests:\t\t\t" << runBurstTests << std::endl;
   outParamStr << "Burst Block Size:\t\t" << burstBlockSize << std::endl;
   outParamStr << "Ranged Tests:\t\t\t" << runRangeTests << std::endl;
   outParamStr << "Sustained Tests:\t\t" << runSustainedTests << std::endl;
   outParamStr << "Test All Sockets:\t\t" << runSocketTests << std::endl;
   outParamStr << "Number Repeated Steps:\t\t" << numStepRepeats << std::endl;
   outParamStr << "Number Steps Per Magnitude:\t" << numRangeSteps << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Memory Overhead Test --------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runMemoryOverheadTest << std::endl;
   outParamStr << "Allocation Range: \t\t";
   outParamStr << rangeMemOverhead[0] << "," << rangeMemOverhead[1] << " (min,max)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Host Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestHH << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHH << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostHostBW[0] << "," << rangeHostHostBW[1] << " (min,max)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "-------------------- Host-Device Bandwidth Test -----------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestHD << std::endl;
   outParamStr << "Run All Memory Patterns :\t" << runPatternsHD << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostDeviceBW[0] << "," << rangeHostDeviceBW[1] << " (min,max)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- P2P Bandwidth Test ----------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Test:\t\t\t" << runBandwidthTestP2P << std::endl;
   outParamStr << "Allocation Range:\t\t";
   outParamStr << rangeDeviceBW[0] << "," << rangeDeviceBW[1] << " (min,max)" << std::endl;
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "----------------------- Pipeline Contention ---------------------" << std::endl; 
   outParamStr << "-----------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Contention Test:\t\t" << runContentionTest << std::endl;
   outParamStr << "# Repeated Operations:\t\t" << numContRepeats << std::endl;
   outParamStr << "Local Host Block Size:\t\t" << contBlockSize[0] << std::endl;
   outParamStr << "Host QPI Mem BlockSize:\t\t" << contBlockSize[1] << std::endl;
   outParamStr << "PCIe Mem Block Size:\t\t" << contBlockSize[2] << std::endl;
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


