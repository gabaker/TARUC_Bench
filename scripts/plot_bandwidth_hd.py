import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

class text:
   bold = '\033[1m'
   italic = '\033[3m'
   blue = '\033[34m'
   red = '\033[91m'
   end = '\033[0m'

# blue, red, green, yellow, orange, purple, aqua, brown, gold, maroon, lime, fushia, dark gray, misty rose, tan, dark khaki, navy, cadet blue, black
color = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#FFA500', '#800080', '#00FFFF', '#A52A2A', '#FFD700', '#800000', '#00FF00', '#FF00FF', '#A9A9A9', '#FFE4E1', '#D2B48C', '#000080', '#BDB76B', '#000080', '#5F9EA0', '#000000']
#marker=list("o^sDx*8.|h15p+_")
marker=list("o^sDx*8.|h1p+_")

class text:
   bold = '\033[1m'
   italic = '\033[3m'
   blue = '\033[34m'
   red = '\033[91m'
   end = '\033[0m'

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
numPatterns = int(testParams[3])
numDirs = 2
startSocket = 0

numMems = 1
if (testParams[4] == "t"):
   numMems = 3

if (testParams[5] != "t"):
   numSockets = 1

memTag = ["page","pin","wc","device"]
memLabel = ["Pageable","Pinned","Write-Combined","Device"]
memLabelShort = ["Page","Pin","WC","Device"]
dirTag = ["h2d","d2h"]
dirLabel = ["Host-to-Device","Device-to-Host"]
dirLabelShort = ["H2D","D2H"]
patternLabel=["Repeated","Linear Inc","Linear Dec"]
patternLabelShort=["Rep","Inc","Dec"]
patternTag=["repeat","linear_inc","linear_dec"]

device = []
for idx in range(0, numDevices):
   device.append(testParams[idx + 6]) 

print ("\nPlotting results from file " + text.italic + text.bold + text.red + "" 
      "" + sys.argv[1] + text.end + " given parameters:")
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "Device Count: " + str(numDevices)
print "# Access Patterns: " + str(numPatterns)
print "Mem Labels: " + str(memLabel)
print "Mem Tags: " + str(memTag)
print "Transfer Labels: " + str(dirLabel)
print "Transfer Tags: " + str(dirTag)
print "Pattern Labels: " + str(patternLabel)
print "Pattern Tags: " + str(patternTag)
print "Devices: " + str(device)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

#set print and save parameters depending on bw or tt type of graphs
xmax = int(blkSize[-1] * 1.2)
ymax = int(np.amax(data) * 1.2)
#xmin = 0
#ymax = 0
ymin = 0
saveType = ""
xscale = 'log'
yscale = 'log'
ylabel = ''
 
if (printBW):
   ylabel = 'Copy Bandwidth (GB/S)'
   yscale = 'linear'
   saveType = "bw"
   ymin = 0.001
else:
   ylabel = 'Transfer Time Per Block (us)'
   saveType = "tt"
   ymin = 0.1

#function for saving specific plot to file
def save_figure(tag, title, large_plot=False):
   plt.figure(tag)
   plt.xscale(xscale)
   plt.yscale(yscale)
   plt.ylim(ymax=ymax)
   plt.ylim(ymin=ymin)
   plt.xlim(xmax=xmax)
   if (large_plot == True):
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.50)
   else:   
      plt.legend(loc='upper left', bbox_to_anchor=(0.0,1.0), fontsize=10, labelspacing=0.50)

   #plt.title(title)
   plt.ylabel(ylabel)
   plt.xlabel('Block Size (bytes)')
   plt.savefig("./bandwidth/hd/" + saveType +"/" + tag + ".png", bbox_inches='tight', dpi=200)
   plt.clf()         
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(tag)
   plt.scatter(x, y, c = color, marker = mark, label = label, linewidth = 0.25, s=12) 
   return

#CASE 0: Each socket, each src/dest pair, each host mem, each direction, all patterns
#CASE 1: Each socket, each src/dest pair, each pattern, each direction, all host mems
#CASE 2: Each socket, each src/dest pair, each patterns, each host mem, both directions
#CASE 3: Each socket, each pattern, each host mem, each direction, all src/dest pairs
#CASE 4: Each socket, each pattern, both direction, all host mems, all src/dest pair
#CASE 5: Each socket, each pattern, each direction, all host mems, all src/dest pair
for socket in range(0, numSockets):
   for hostIdx in range(0, numNodes):
      for devIdx in range(0, numDevices):
         for memIdx in range(0, numMems):
            for dirIdx in range(0, numDirs):
               for patternIdx in range(0, numPatterns):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
                  
                  #CASE 0: Each socket, each src/dest pair, each host mem, each direction, all patterns
                  tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + memTag[memIdx] + "_" + dirTag[dirIdx] + "_all_patterns"
                  label = patternLabel[patternIdx]
                  add_scatter(blkSize, data[idx], color[patternIdx], marker[patternIdx], tag, label)

               #CASE 0: Each socket, each src/dest pair, each host mem, each direction, all patterns
               tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + memTag[memIdx] + "_" + dirTag[dirIdx] + "_all_patterns"
               save_figure(tag, tag)
         
         for patternIdx in range(0, numPatterns):
            for dirIdx in range(0, numDirs):
               for memIdx in range(0, numMems):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
 
                  #CASE 1: Each socket, each src/dest pair, each pattern, each direction, all host mems
                  tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_all_host_mems"
                  label = memLabel[memIdx] 
                  add_scatter(blkSize, data[idx], color[memIdx], marker[memIdx], tag, label)
               
               #CASE 1: Each socket, each src/dest pair, each pattern, each direction, all host mems
               tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_all_host_mems"
               save_figure(tag, tag)

            for memIdx in range(0, numMems):
               for dirIdx in range(0, numDirs):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
 
                  #CASE 2: Each socket, each src/dest pair, each patterns, each host mem, both directions
                  tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + memTag[memIdx] + "_" + patternTag[patternIdx] + "_both_dirs"
                  label = dirLabel[dirIdx]
                  add_scatter(blkSize, data[idx], color[dirIdx], marker[dirIdx], tag, label)
               
               #CASE 2: Each socket, each src/dest pair, each patterns, each host mem, both directions
               tag = "cpu" + str(socket) + "_host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + memTag[memIdx] + "_" + patternTag[patternIdx] + "_both_dirs"
               save_figure(tag, tag)

   for patternIdx in range(0, numPatterns):
      for memIdx in range(0, numMems):
         for dirIdx in range(0, numDirs):
            for hostIdx in range(0, numNodes):
               for devIdx in range(0, numDevices):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
                  
                  #CASE 3: Each socket, each pattern, each host mem, each direction, all src/dest pairs
                  tag = "cpu" + str(socket) + "_" + patternTag[patternIdx] + "_" + memTag[memIdx] + "_" + dirTag[dirIdx] + "_all_host_dev"
                  label = "Node: " + str(hostIdx) + " " + device[devIdx]
                  colorIdx = (devIdx * numNodes + hostIdx) % len(color) 
                  add_scatter(blkSize, data[idx], color[colorIdx], marker[hostIdx], tag, label)
                   
                  #CASE 4: Each socket, each pattern, both direction, all host mems, all src/dest pair
                  tag = "cpu" + str(socket) + "_" +  patternTag[patternIdx] + "_all_host_dev_dirs_mems"
                  label = "Node " + str(hostIdx) + " " + device[devIdx] + " " + dirLabelShort[dirIdx] + " " + memLabel[memIdx]
                  colorIdx = (devIdx * numNodes + hostIdx) % len(color)
                  add_scatter(blkSize, data[idx], color[colorIdx], marker[dirIdx * numMems + memIdx], tag, label)

            #CASE 3: Each socket, each pattern, each host mem, each direction, all src/dest pairs
            tag = "cpu" + str(socket) + "_" + patternTag[patternIdx] + "_" + memTag[memIdx] + "_" + dirTag[dirIdx] + "_all_host_dev"
            
            if (numDevices > 2):
               save_figure(tag, tag, True)
            else:
               save_figure(tag, tag)
                           

      #CASE 4: Each socket, each pattern, both direction, all host mems, all src/dest pair
      tag = "cpu" + str(socket) + "_" +  patternTag[patternIdx] + "_all_host_dev_dirs_mems"
      save_figure(tag, tag, True)
   
      for dirIdx in range(0, numDirs):
         for memIdx in range(0, numMems):
            for hostIdx in range(0, numNodes):
               for devIdx in range(0, numDevices):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
 
                  #CASE 5: Each socket, each pattern, each direction, all host mems, all src/dest pair
                  tag = "cpu" + str(socket) + "_" +  patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_all_host_dev_mems"
                  label = "Node " + str(hostIdx) + " " + device[devIdx] + " " + memLabel[memIdx]
                  colorIdx = (devIdx * numNodes + hostIdx) % len(color)
                  add_scatter(blkSize, data[idx], color[colorIdx], marker[memIdx], tag, label)

         #CASE 5: Each socket, each pattern, each direction, all host mems, all src/dest pair
         tag = "cpu" + str(socket) + "_" +  patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_all_host_dev_mems"
         save_figure(tag, tag, True)

#CASE 6: Each src/dest pair, each pattern, each direction, each host mem, all sockets
#CASE 7: Each src/dest pair, each pattern, both directions, all host mems, all sockets
for hostIdx in range(0, numNodes):
   for devIdx in range(0, numDevices):
      for patternIdx in range(0, numPatterns):
         for dirIdx in range(0, numDirs):
            for memIdx in range(0, numMems):
               for socket in range(0, numSockets):
                  idx = socket * (numNodes * numDevices * numMems * numDirs * numPatterns) + \
                        hostIdx * (numDevices * numMems * numDirs * numPatterns) + \
                        devIdx * (numMems * numDirs * numPatterns) + \
                        memIdx * (numDirs * numPatterns) + \
                        dirIdx * (numPatterns) + patternIdx
 
                  #CASE 6: Each src/dest pair, each pattern, each direction, each host mem, all sockets
                  tag = "host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_" + memTag[memIdx] + "_all_cpus"
                  label = "CPU " + str(socket)
                  colorIdx = socket % len(color)
                  add_scatter(blkSize, data[idx], color[colorIdx], marker[socket], tag, label)

                  #CASE 7: Each src/dest pair, each pattern, both directions, all host mems, all sockets
                  tag = "host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_all_cpus_dirs_mems"
                  label = "CPU " + str(socket) + " " + dirLabelShort[dirIdx] + " " + memLabel[memIdx]
                  colorIdx = (socket * numMems * numDirs + memIdx * numDirs + dirIdx) % len(color) 
                  add_scatter(blkSize, data[idx], color[colorIdx], marker[socket * numDirs + dirIdx], tag, label)
                  
               #CASE 6: Each src/dest pair, each pattern, each direction, each host mem, all sockets
               tag = "host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_" + dirTag[dirIdx] + "_" + memTag[memIdx] + "_all_cpus"
               save_figure(tag, tag)

         #CASE 7: Each src/dest pair, each pattern, both directions, all host mems, all sockets
         tag = "host" + str(hostIdx) + "_dev" + str(devIdx) + "_" + patternTag[patternIdx] + "_all_cpus_dirs_mems"
         save_figure(tag, tag, True)


