import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = list("brygcm")

if (len(sys.argv) < 3):
   print "Usage: python plot_bandwidth.py results_file.csv topo_file.out [parameter_file.out]"
   sys.exit() 

if (os.path.isfile(sys.argv[2]) == False):
   print "Failed to open file: " + sys.argv[2]
   sys.exit()

topo = open(sys.argv[2])

numDevices = 0 
numNumaNodes = 0
numSockets = 0
for line in topo:
   line = line.strip()

   if ("NUMA Nodes:" in line):
      line = line.replace("\t", "").split(":")
      numNumaNodes = int(line[1])
   if ("Sockets:" in line):
      line = line.replace("\t", "").split(":")
      numSockets = int(line[1])
   if ("Accelerators:" in line):
      line = line.replace("\t", "").split(":")
      numDevices = int(line[1])

deviceNames=[numDevices]
runAllDevices = False
usePinnedMem = False
runBandwidthTest = False
runRangeTest = False
runAllPatterns = False

#read relevent parameters from output benchmark parameters file
if (len(sys.argv) == 4):
   if (os.path.isfile(sys.argv[3]) == False):
      print "Failed to open file: " + sys.argv[2]
      sys.exit()
   params = open(sys.argv[3])
   paramStr = params.read()

   idxNext = paramStr.find("inned")
   idx = paramStr.find("evices")
   if ( paramStr.find("rue", idx, idxNext) > 0):
      runAllDevices=True

   idx = paramStr.find("inned")
   idx = paramStr.find("Mem", idx)
   if ( paramStr.find("rue", idx, idx + 10) > 0):
      usePinnedMem=True

   idx = paramStr.find("Bandwidth", idx)
   idx = paramStr.find("Bandwidth", idx + 1)
   idx = paramStr.find("Test", idx)
   if ( paramStr.find("rue", idx, idx + 10) > 0):
      runBandwidthTest=True
 
   idx = paramStr.find("Range", idx)
   idxNext = paramStr.find("Burst", idx)
   if ( paramStr.find("rue", idx, idxNext) > 0):
      runRangedTest=True

   idx = paramStr.find("Patterns")
   idx = paramStr.find("rns", idx)   
   if ( paramStr.find("rue",idx, idx + 10) > 0):
      runAllPatterns=True

print "\nPlotting bandwidth results from file " + sys.argv[2] + "given parameters:"
print "Node Count: " + str(numNumaNodes)
print "Socket Count: " + str(numSockets)
print "Device Count: " + str(numDevices)
print "Bandwidth Test: " + str(runBandwidthTest)
print "Range Test: " + str(runRangedTest)
print "Use Pinned Memory: " + str(usePinnedMem)
print "Use All Devices: " + str(runAllDevices)
print "All Patterns: " + str(runAllPatterns)

numPatterns = 1
if (runAllPatterns == True):
   numPatterns = 4

numMemTypes = 1
if (usePinnedMem == True):
   numMemTypes = 2

numCols = (numDevices + numMemTypes + numNumaNodes + (numMemTypes - 2 + 1)) * numNumaNodes * numPatterns 

if ((runRangedTest and runBandwidthTest) == False):
   print "\nParameters stated no ranged Bandwidth test...exiting\n"
   sys.exit()

# read each column into the correct location, in order
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0))
bandData = []

#maxMag = math.floor(math.log10(blkSize[-1]))
xmax = int(blkSize[-1] * 1.5)
print "" + str(xmax)

for idx in range(0, numCols):
   bandData.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx)))

#function for saving specific plot to file
def save_figure( figureNum, title, saveName ):
   plt.figure(figureNum)
   plt.xscale('log')
   plt.yscale('log')
   plt.ylim(ymin=0.0001)
   plt.xlim(xmax=5000000000)
   plt.legend(loc='upper left', fontsize=8)

   plt.title(title)
   plt.ylabel('Call Duration (ms)')
   plt.xlabel('Freed Block Size (bytes)')

   plt.savefig("./results/" + saveName + ".png", bbox_inches='tight')
   return

# function for clearing all plots between graphing runs
def clear_plot_range( begin, end):
   for plotIdx in range(begin, end + 1):
      plt.figure(plotIdx)
      plt.clf()
   return

#CASE 0: Host to Host, Pageable host memory
#CASE 1: Host to Host, Pinned source memory, Pageable dest memory
#CASE 2: Host to Host, Pageable source memory, Pinned dest memory
#CASE 3: Host to Host, Pageable source and dest memory

#CASE 4: Host to Device, Pageable host memory
#CASE 5: Host to Device, Pinned host memory

#CASE 6: Device to Host, Pageable host memory 
#CASE 7: Device to Host, Pinned host memory

#change range end to numSockets when all cpus are being tested
for socket in range(0, 1):
   for srcNode in range(0, numNumaNodes):
      for destNode in range(0, numNumaNodes):
         label = "Src Node:" + str(srcNode) + "Dest:" + str(srcNode)
         
         for pattern in range(0, numPatterns):
            #CASE 0: Host to Host, Pageable host memory
            plt.figure("" + str(0))
            plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
            plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

        
            #CASE 1: Host to Host, Pinned source memory, Pageable dest memory
         
            #CASE 2: Host to Host, Pageable source memory, Pinned dest memory
         
            #CASE 3: Host to Host, Pageable source and dest memory

        
         #CASE 0:
         plt.figure("" + str(0))
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

         #CASE 1: Host to Host, Pinned source memory, Pageable dest memory
      
      for dev in range(0, numDevices):
         #label = 
         #devIdx =  
         
         #CASE 0:
         plt.figure("alloc" + str(0)) 
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.figure("free" + str(0)) 
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 

#CASE 0
save_figure("alloc" + str(0), "", "alloc_all_numa_cpu_dev")
save_figure("free" + str(0), "", "free_all_numa_cpu_dev")

plt.show()

