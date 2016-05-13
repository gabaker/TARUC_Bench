import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = list("brygcm")

if (len(sys.argv) < 2):
   print "Usage: python script_name.py results_file.csv"
   sys.exit() 

if (os.path.isfile(sys.argv[1]) == False):
   print "Failed to open file: " + sys.argv[1]
   sys.exit()

#check if printing bandwidth or transfer time graphs
printBW = False
if (sys.argv[1].find("bw") != -1):
   printBW = True

# read in results file header values
results = open(sys.argv[1])
testParams = results.readline().strip().split(",");

numSockets = int(testParams[0])
numDevices = int(testParams[1])
numGroups = int(testParams[2])
numTransTypes = 2
#numTransTypes = 4

useSockets = False
if (testParams[3] == "t"):
   useSockets = True
else:
   startSocket = numSockets
   numSockets = 1

transTypes = ["d2d","peer", "d2d_uva", "peer_uva"]
devices=[numDevices]

devices = []
for idx in range(0, numDevices):
   devices.append(testParams[idx + 4]) 

uvaList = []
for group in range(0, numDevices):
   uvaList.append(testParams[idx + 4 + numDevices]) 

peerList = []
for group in range(0, numGroups):
   peerList.append(results.readline().strip().split(","))
totalTrans = 0
numTransPerPair = [[0 for x in range(numDevices + 1)] for x in range(numDevices)]
for srcDev in range(0, numDevices):
   uva = False
   if (uvaList[srcDev] == "true"):
      uva = True
   for destDev in range(0, numDevices):
      numTransPerPair[srcDev][destDev] = numTransPerPair[srcDev][max(0, destDev - 1)] + 1
      if (uva):
         numTransPerPair[srcDev][destDev] += 1
      for group in peerList:
         if ( srcDev != destDev and str(destDev) in group and str(srcDev) in group):
            numTransPerPair[srcDev][destDev] += 1
            if (uva): 
               numTransPerPair[srcDev][destDev] += 1

   numTransPerPair[srcDev][numDevices] = numTransPerPair[srcDev][numDevices - 1] + numTransPerPair[max(0, srcDev - 1)][numDevices]

print "\nPlotting P2P throughput/bandwidth results from file " + sys.argv[1] + "given parameters:"
print "Socket Count: " + str(numSockets)
print "Device Count: " + str(numDevices)
print "Num Peer Groups: " + str(numGroups)
print "Use All Sockets: " + str(useSockets)
print "Transfer Types: " + str(transTypes)
print "Devices: " + str(devices)
print "Peer Groups: " + str(peerList)
print "Transfer Count" + str(numTransPerPair)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1 + numGroups))

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1 + numGroups)))

#set print and save parameters depending on bw or tt type of graphs
plot_xmax = int(blkSize[-1] * 1.5)
plot_ymin = 0
saveType = ""
xscale = 'log'
yscale = 'log'
ylabel = ''
 
if (printBW):
   ylabel = 'Copy Bandwidth (GB/S)'
   xscale = 'log'
   yscale = 'linear'
   saveType = "p2p_bw_"
   plot_ymin = 0#0.001
else:
   ylabel = 'Transfer Time Per Block (ms)'
   xscale = 'log'
   yscale = 'linear'
   saveType = "p2p_tt_"
   plot_ymin = 0

#function for saving specific plot to file
def save_figure(figTag, title, saveName):
   plt.figure(figTag)
   plt.xscale(xscale)
   plt.yscale(yscale)
   plt.ylim(ymin=plot_ymin)
   plt.xlim(xmax=plot_xmax)
   plt.legend(loc='upper left', fontsize=8)

   plt.title(title)
   plt.ylabel(ylabel)
   plt.xlabel('Block Size (bytes)')

   plt.savefig("./results/" + saveType + saveName + ".png", bbox_inches='tight')
   return

# CASE 8: All Cross Socket Transfers

# CASE 0: Each Socket, Each D2D Pair, Each Direction, All Transfer Types
# CASE 1: Each Socket, Each D2D Pair, Both Direction, All Transfer Types
# CASE 2: Each Socket, All to One D2D Set, Each Transfer Type, each direction
# CASE 3: Each Socket, All to One D2D Set, All Transfer Types, each direction  
# CASE 4: Each Socket, One to All D2D Set, Each Transfer Type, each direction
# CASE 5: Each Socket, One to All D2D Set, All Transfer Types, each direction

totalTrans = numTransPerPair[numDevices - 1][numDevices]
idx = 0
for socket in range(0, numSockets):
   for srcDev in range(0, numDevices):
      for destDev in range(0, numDevices):
         uva = False
         peer = False
         if (uvaList[srcDev] == "true"):
            uva = True
         for group in peerList:
            if ( srcDev != destDev and str(destDev) in group and str(srcDev) in group):
               peer = True
         
         idx = socket * totalTrans + numTransPerPair[srcDev][destDev] 

         # CASE 0
         # CASE 1


# CASE 6: Each transfer type, Each Direction, Each Socket, All D2D Pairs, 
# CASE 7: Each transfer type, Each Direction, All Socket, All D2D Pairs
for transType in range(0, numTransTypes):
   for srcDev in range(0, numDevices):
      for destDev in range(0, numDevices):
         for socket in range(0, numSockets):
            idx = socket * 1 
      
   for socket in range(0, numSockets):
      for destDev in range(0, numDevices):
         for srcDev in range(0, numDevices):
            idx = socket * 1 



'''         y_idx =  socket * numSockets * numNodes * numMemGroups * numPatterns + srcNode * numNodes * numMemGroups * numPatterns + destNode * numMemGroups * numPatterns
         
         #CASE 0: Each socket, all mem combinations, each src/dest pair, one pattern
         label = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_mem_types"
         plt.figure(label)
         plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[0], label = "Both Page") 
         if (usePinnedMem): 
            plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[1], label = "Pinned Src") 
            plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[2], label = "Pinned Dest") 
            plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[3], label = "Both Pinned") 

         save_figure(label, "H2H: CPU " + str(socket) + " Src " + str(srcNode)+ " Dest " + str(destNode) + ", All Mem Types", label)
         plt.clf()         
'''
































 
