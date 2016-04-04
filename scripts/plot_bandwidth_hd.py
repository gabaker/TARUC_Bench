import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#800000', '#C0C0C0', '#800080', '#000000', '#00FFFF', '#A5522D']
marker=list("o^sDx")
#"", 
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

# read in results file header
results = open(sys.argv[1])
testParams = results.readline().strip().split(",");

numSockets = int(testParams[0])
numNodes = int(testParams[1])
numDevices = int(testParams[2])
numDirs = 2
numMems = 1
startSocket = 0

usePinnedMem = False
if (testParams[3] == "t"):
   usePinnedMem = True
   numMems = 2

useSockets = False
if (testParams[4] == "t"):
   useSockets = True
else:
   startSocket = numSockets
   numSockets = 1

memTypes = ["page","pin"]
transTypes = ["h2d","d2h"]
devices = []
for idx in range(0, numDevices):
   devices.append(testParams[idx + 5]) 

patterns = []
numPatterns = len(testParams) - numDevices - 5
for idx in range(0, numPatterns):
   patterns.append(testParams[idx + numDevices + 5])

print "\nPlotting Host-Device bandwidth results from file " + sys.argv[1] + " given parameters:"
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "Device Count: " + str(numDevices)
print "Use Pinned Memory: " + str(usePinnedMem)
print "Use All Sockets: " + str(useSockets)
print "Mem Types: " + str(memTypes)
print "Transfer Types: " + str(transTypes)
print "Devices: " + str(devices)
print "Patterns: " + str(patterns)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))

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
   saveType = "hd_bw_"
   plot_ymin = 0.001
else:
   ylabel = 'Transfer Time Per Block (ms)'
   xscale = 'log'
   yscale = 'linear'
   saveType = "hd_tt_"
   plot_ymin = 0

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

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

#CASE 1: Each socket, each src/dest pair, each pattern, each direction, both mem combination
#CASE 2: Each socket, each src/dest pair, each patterns, each mem combination, both directions
#CASE 3: Each socket, each pattern, each mem combination, each direction, all src/dest pairs
#CASE 4: Each socket, each pattern, each mem combination, all src/dest pair, both direction, both mem combination
#CASE 5: Each src/dest pair, each pattern, each direction, each mem combination, all sockets
#CASE 6: Each src/dest pair, each pattern, both directions, both mem combination, all sockets

for socket in range(startSocket, startSocket + numSockets):
   for hostIdx in range(0, numNodes):
      for devIdx in range(0, numDevices):
         for memory in range(0, numMems):
            for transDir in range(0, numDirs):
               #CASE 0: Each socket, each src/dest pair, each mem combination, each direction, all patterns
               for pattern in range(0, numPatterns):
                  #CASE 0: All patterns
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_all_patterns"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[pattern], label = patterns[pattern]) 

               #CASE 0: All patterns
               label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_all_patterns"
               save_figure(label, label, label)
               plt.clf()         

           
         for pattern in range(0, numPatterns):
            #CASE 1: Each socket, each src/dest pair, each pattern, each direction, both mem combination
            for transDir in range(0, numDirs):
               for memory in range(0, numMems):
                  #CASE 1
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + patterns[pattern] + "_both_mems"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[memory], label = patterns[memory], marker = marker[memory]) 
               #CASE 1
               label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + patterns[pattern] + "_both_mems"
               save_figure(label, label, label)
               plt.clf()         

            #CASE 2: Each socket, each src/dest pair, each patterns, each mem combination, both directions
            for memory in range(0, numMems):
               for transDir in range(0, numDirs):
                  #CASE 2
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + memTypes[memory] + "_" + patterns[pattern] + "_both_dirs"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[transDir], label = transTypes[transDir], marker = marker[transDir]) 
               #CASE 2
               label = "cpu" + str(socket) + "_src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + memTypes[memory] + "_" + patterns[pattern] + "_both_dirs"
               save_figure(label, label, label)
               plt.clf()


   for pattern in range(0, numPatterns):
      for memory in range(0, numMems):
         #CASE 3: Each socket, each pattern, each mem combination, each direction, all src/dest pairs
         for transDir in range(0, numDirs):
            for hostIdx in range(0, numNodes):
               for devIdx in range(0, numDevices):
                  #CASE 3
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "cpu" + str(socket) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_" + patterns[pattern] + "_all_src_dest"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[hostIdx * numNodes + devIdx], label = "" + "Host " + str(hostIdx) +" " + devices[devIdx], marker = marker[memory]) 
               #CASE 3
               label = "cpu" + str(socket) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_" + patterns[pattern] + "_all_src_dest"
               save_figure(label, label, label)
               plt.clf()

         #CASE 4: Each socket, each pattern, each mem combination, all src/dest pair, both direction, both mem combination
         for hostIdx in range(0, numNodes):
            for devIdx in range(0, numDevices):
               for transDir in range(0, numDirs):
                  #CASE 4
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "cpu" + str(socket) + "_" +  memTypes[memory] + "_" + patterns[pattern] + "_all_src_dest_dirs_mems"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[hostIdx * numNodes + devIdx], label = transTypes[transDir] + ", " + memTypes[memory] + ", Host " + str(hostIdx) + ", " + devices[devIdx], marker = marker[memory + transDir * numMems]) 
               #CASE 4
               label = "cpu" + str(socket) + "_" +  memTypes[memory] + "_" + patterns[pattern] + "_all_src_dest_dirs_mems"
               save_figure(label, label, label)
               plt.clf()
 
for hostIdx in range(0, numNodes):
   for devIdx in range(0, numDevices):
      for pattern in range(0, numPatterns):
         #CASE 5: Each src/dest pair, each pattern, both directions, both mem combination, all sockets
         for transDir in range(0, numDirs):
            for memory in range(0, numMems):
               #CASE 6: Each src/dest pair, each pattern, each direction, each mem combination, all sockets
               for socket in range(startSocket, startSocket + numSockets):
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_" + patterns[pattern] + "_all_cpus"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[socket], label = "CPU " + str(socket), marker = marker[socket]) 

                  # CASE 5
                  y_idx = socket * numNodes * numDevices * numMems * numDirs * numPatterns + hostIdx * numDevices * numMems * numDirs * numPatterns + devIdx * numMems * numDirs * numPatterns + memory * numDirs * numPatterns + transDir * numPatterns + pattern 
                  label = "src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + patterns[pattern] + "_all_cpus_dirs_mems"
                  plt.figure(label)
                  plt.scatter(blkSize, data[y_idx], c = colors[transDir * numMems + memory], label = transTypes[transDir] + ", " + memTypes[memory] + ", CPU " + str(socket), marker = marker[socket]) 

               # CASE 6
               label = "src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + transTypes[transDir] + "_" + memTypes[memory] + "_" + patterns[pattern] + "_all_cpus"
               save_figure(label, label, label)
               plt.clf()
         
         # CASE 5
         label = "src" + str(hostIdx) + "_dest" + str(devIdx) + "_" + patterns[pattern] + "_all_cpus_dirs_mems"
         save_figure(label, label, label)
         plt.clf()


           
 



         





















             












