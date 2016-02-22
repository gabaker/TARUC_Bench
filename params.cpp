
//#ifndef PARAMS_CLASS_INC
//#define PARAMS_CLASS_INC
#include "params.h"
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

   GetNextLine(inFile, lineStr); //resultsFile
   resultsFile = lineStr.substr(lineStr.find ('=') + 1);

   printDevProps = GetNextLineBool(inFile, lineStr); //printDeviceProps
   GetNextLine(inFile, lineStr);
   devPropFile = lineStr.substr(lineStr.find ('=') + 1); //devPropFile
   GetNextLine(inFile, lineStr);
   topoFile = lineStr.substr(lineStr.find ('=') + 1); //topoFile
   runTopoAware = GetNextLineBool(inFile, lineStr); //runTopoAware
  
   runMemoryOverheadTest = GetNextLineBool(inFile, lineStr); //runMemoryOverheadTest
   runAllDevices = GetNextLineBool(inFile, lineStr); //runAllDevices 
   for (int i = 0; i < 3; i++) {
      GetNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      rangeMemOverhead[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }

   runHostDeviceBandwidthTest = GetNextLineBool(inFile, lineStr); //runHostDeviceBandwidthTest
   varyBlockSizeHD = GetNextLineBool(inFile, lineStr); //varyBlockSizeHD
   usePinnedHD = GetNextLineBool(inFile, lineStr); //usePinnedHD
   runBurstHD = GetNextLineBool(inFile, lineStr); //runBurstHD
   runSustainedHD = GetNextLineBool(inFile, lineStr); //runSustainedHD
   for (int i = 0; i < 3; i++) {
      GetNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      rangeHostDeviceBW[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }

   runP2PBandwidthTest = GetNextLineBool(inFile, lineStr); //runP2PBandwidthTest
   varyBlockSizeP2P = GetNextLineBool(inFile, lineStr); //varyBlockSizeP2P
   runBurstP2P = GetNextLineBool(inFile, lineStr); //runBurstHD
   runSustainedP2P = GetNextLineBool(inFile, lineStr); //runSustainedHD
   for (int i = 0; i < 3; i++) {
      GetNextLine(inFile, lineStr);
      int eqIdx = lineStr.find("=") + 1;
      rangeDeviceP2P[i] = std::atol(lineStr.substr(eqIdx).c_str());
   }
   
   runPCIeCongestionTest = GetNextLineBool(inFile, lineStr); //runPCIeCongestionTest
   runTaskScalabilityTest = GetNextLineBool(inFile, lineStr); //runTaskScalabilityTest
   
}

// Set default device properties based on an interesting variety of tests 
// in case no input file is provided. These values do necessarily reflect 
// what the developer recommends to demonstrate category performance on any 
// specific system system
void BenchParams::SetDefault() {

   resultsFile = "results";
   inputFile = "none";
   useDefaultParams = true;

   printDevProps = true;
   devPropFile = "device_info.out";
   topoFile = "none";
   runTopoAware = false;

   runMemoryOverheadTest = true; 
   runAllDevices = true;
   rangeMemOverhead[0] = 1;
   rangeMemOverhead[1] = 1000001;
   rangeMemOverhead[2] = 10;
   
   runHostDeviceBandwidthTest = false;
   varyBlockSizeHD = true;
   usePinnedHD = true;
   runBurstHD  = true;
   runSustainedHD = true;
   rangeHostDeviceBW[0] = 1;
   rangeHostDeviceBW[1] = 1024;
   rangeHostDeviceBW[2] = 2; 
  
   runP2PBandwidthTest = false;
   varyBlockSizeP2P = true;
   runBurstP2P = true;
   runSustainedP2P = true;
   rangeDeviceP2P[0] = 1;
   rangeDeviceP2P[1] = 2024;
   rangeDeviceP2P[2] = 2;
   
   runPCIeCongestionTest = false;
   
   runTaskScalabilityTest = false;
}

void BenchParams::PrintParams() {
   std::stringstream outParamStr;

   outParamStr << std::boolalpha;
   outParamStr << "\n------------------------------------------------------------" << std::endl; 
   outParamStr << "---------------------- Test Parameters ---------------------" << std::endl; 
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Input File:\t\t\t" << inputFile << std::endl;
   outParamStr << "Output file:\t\t\t" << resultsFile << std::endl;
   outParamStr << "Using Defaults:\t\t\t" << useDefaultParams << std::endl;  
   outParamStr << "Printing Device Props:\t\t" << printDevProps << std::endl;
   outParamStr << "Device Property File:\t\t" << devPropFile << std::endl;
   outParamStr << "Topology File:\t\t\t" << topoFile << std::endl;  
   outParamStr << "Running topology aware:\t\t" << runTopoAware << std::endl;
   outParamStr << "Device Count:\t\t\t" << nDevices << std::endl;
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Memory Overhead Test:\t" << runMemoryOverheadTest << std::endl;
   outParamStr << "Use all Devices:\t\t" << runAllDevices << std::endl;
   outParamStr << "Allocation Range: \t\t";
   outParamStr << rangeMemOverhead[0] << "," << rangeMemOverhead[1];
   outParamStr << "," << rangeMemOverhead[2] << " (min,max,step)" << std::endl;
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Host-Device Bandwidth Test:\t" << runHostDeviceBandwidthTest << std::endl;
   outParamStr << "Vary Block Size:\t\t" << varyBlockSizeHD << std::endl;
   outParamStr << "Use Pinned Host Mem:\t\t" << usePinnedHD << std::endl;
   outParamStr << "Burst Mode:\t\t\t" << runBurstHD << std::endl;
   outParamStr << "Sustained Mode:\t\t\t" << runSustainedHD << std::endl;
   outParamStr << "Allocation Range:\t\t"; 
   outParamStr << rangeHostDeviceBW[0] << "," << rangeHostDeviceBW[1] << ","; 
   outParamStr << rangeHostDeviceBW[2] << " (min,max,step)" << std::endl;
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Run P2P Bandwidth Test:\t\t" << runP2PBandwidthTest << std::endl;
   outParamStr << "Vary Block Size:\t\t" << varyBlockSizeP2P << std::endl;
   outParamStr << "Burst Mode:\t\t\t" << runBurstP2P << std::endl;
   outParamStr << "Sustained Mode:\t\t\t" << runSustainedP2P << std::endl;
   outParamStr << "Allocation Range:\t\t";
   outParamStr << rangeDeviceP2P[0] << "," << rangeDeviceP2P[1] << ",";
   outParamStr << rangeDeviceP2P[2] << " (min,max,step)" << std::endl;
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Run PCIe CongestionTest:\t" << runPCIeCongestionTest << std::endl;
   outParamStr << "------------------------------------------------------------" << std::endl; 
   outParamStr << "Run Task Scalability Test:\t" << runTaskScalabilityTest << std::endl; 
   outParamStr << "------------------------------------------------------------\n" << std::endl;    
   outParamStr << std::noboolalpha;

   //Print benchmark parameters to string
   std::cout << outParamStr.str();

   // Print benchmark parameters to output file
   std::string paramFileName = "bench_params.out";
   std::ofstream outParamFile(paramFileName.c_str());
   outParamFile << outParamStr << std::endl;

   //read params out to command line
   outParamFile.close();
}
