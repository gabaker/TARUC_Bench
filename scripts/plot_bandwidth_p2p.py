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
marker=list("o^sDx*8.|h1p+_")

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

if (testParams[3] != "t"):
   numSockets = 1

numTransTypes = 2
transTag = ["d2d","peer"]
transLabel = ["Device-to-Device Copy","Peer Async Copy"]
transLabel = ["D2D","P2P"]

device = []
for idx in range(0, numDevices):
   device.append(testParams[idx + 4]) 

peerList = []
for group in range(0, numGroups):
   peerList.append(results.readline().strip().split(","))
totalTrans = 0
numTransPerPair = [[0 for x in range(numDevices)] for x in range(numDevices)]

for srcDev in range(0, numDevices):
   for destDev in range(0, numDevices):
      numTransPerPair[srcDev][destDev] = 1
      for group in peerList:
         if (destDev != srcDev and str(destDev) in group and str(srcDev) in group):
            numTransPerPair[srcDev][destDev] += 1

print ("\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + ""
      "" + text.end + " given parameters:")
print "Socket Count: " + str(numSockets)
print "Device Count: " + str(numDevices)
print "Peer Group Count: " + str(numGroups)
print "Peer Groups: " + str(peerList)
print "Transfer Count: " + str(numTransPerPair)
print "Transfer Types: " + str(transLabel)
print "Transfer Tags: " + str(transTag)
print "Devices: " + str(device)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1 + numGroups))

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1 + numGroups)))

#set print and save parameters depending on bw or tt type of graphs
xmax = int(blkSize[-1] * 1.2)
xmin = 0
ymin = 0
ymax = 0
saveType = ""
xscale = ''
yscale = ''
 
if (printBW):
   ylabel = 'Copy Bandwidth (GB/S)'
   yscale = 'linear'
   xscale = 'log'
   saveType = "bw"
else:
   ylabel = 'Transfer Time Per Block (ms)'
   yscale = 'log'
   xscale = 'log'
   saveType = "tt"

#function for saving specific plot to file
def save_figure(tag, title):
   plt.figure(tag)
   plt.xscale(xscale)
   plt.yscale(yscale)
   plt.ylim(ymin=ymin)
   #plt.ylim(ymax=ymax)
   #plt.xlim(xmin=xmin)
   plt.xlim(xmax=xmax)
 
   plt.title(title)
   plt.ylabel(ylabel)
   plt.xlabel('Block Size (bytes)')
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.50)
   plt.savefig("bandwidth/p2p/" + saveType + "/" + tag + ".png", bbox_inches='tight', dpi=150)
   plt.clf()
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(tag)
   plt.scatter(x, y, c = color, marker = mark, label = label, linewidth = 0.25) 
   return

numDirs = 2
prevIdx = 0
# CASE 0: All Non-Intra GPU Transfers
# CASE 1: Each Socket, Each D2D Pair, Both Direction, All Transfer Types
for socket in range(0, numSockets):
   for srcDev in range(0, numDevices):
      for destDev in range(srcDev, numDevices):     
         if (srcDev == destDev):
            numDirs = 1
         else:
            numDirs = 2

         # CASE 1: Each Socket, Each D2D Pair, Both Direction, All Transfer Types
         for transIdx in range(0, numTransPerPair[srcDev][destDev]):
            for dirIdx in range(0, numDirs):
               idx = prevIdx + transIdx * numDirs + dirIdx
               # CASE 0: All Non-Intra GPU Transfers
               if (srcDev != destDev):
                  tag = "all_transfers_no_intra_gpu"
                  if (dirIdx == 0):
                     label = "CPU " + str(socket) + " " + device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
                  else:
                     label = "CPU " + str(socket) + " " + device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
                  add_scatter(blkSize, data[idx], color[socket * 2 + dirIdx], marker[socket], tag, label) 

               tag = "cpu" + str(socket) + "_dev" + str(srcDev) + "_dev" + str(destDev) + "_both_dir_all_trans_types"
               if (dirIdx == 0):
                  label = device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
               else:
                  label = device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
               
               add_scatter(blkSize, data[idx], color[dirIdx], marker[transIdx], tag, label)

         prevIdx += numDirs * numTransPerPair[srcDev][destDev]
         
         # CASE 1: Each Socket, Each D2D Pair, Both Direction, All Transfer Types
         tag = "cpu" + str(socket) + "_dev" + str(srcDev) + "_dev" + str(destDev) + "_both_dir_all_trans_types"
         save_figure(tag, tag)

# CASE 0: All Non-Intra GPU Transfers
tag = "all_transfers_no_intra_gpu"
save_figure(tag, tag)

# CASE 2: Each transfer type, Both Direction, Each Socket, All D2D Pairs 
# CASE 3: Each transfer type, Both Directions, All Sockets, All D2D Pairs
for transIdx in range(0, numTransTypes):
   if (transIdx == 0):
      prevIdx = 0
   else:
      prevIdx = 1
   colorIdx2 = 0
   for socket in range(0, numSockets):
      colorIdx = 0
      for srcDev in range(0, numDevices):
         for destDev in range(srcDev, numDevices):
            if (srcDev == destDev):
               numDirs = 1
            else:
               numDirs = 2

            for dirIdx in range(0, numDirs):
               if (transIdx == 0):
                  # CASE 2: Each transfer type, Each Socket, Both Directions, All D2D Pairs 
                  idx = dirIdx * numTransPerPair[srcDev][destDev]
                  tag = "cpu" + str(socket) + "_" + transTag[transIdx] + "_all_dev_dirs"
                  if (dirIdx == 0):
                     label = device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
                  else:
                     label = device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
                  add_scatter(blkSize, data[idx + prevIdx], color[colorIdx], marker[(destDev * 2 + dirIdx) % len(marker)], tag, label)
                  # CASE 3: Each transfer type, Both Directions, All Sockets, All D2D Pairs
                  tag = transTag[transIdx] + "_all_cpu_dev_dirs"
                  if (dirIdx == 0):
                     label = "CPU " + str(socket) + " " + device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
                  else:
                     label = "CPU " + str(socket) + " " + device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
                  add_scatter(blkSize, data[idx + prevIdx], color[colorIdx2], marker[dirIdx * numSockets + socket], tag, label)

                  colorIdx+=1
               elif (numTransPerPair[srcDev][destDev] == 2):
                  # CASE 2: Each transfer type, Each Socket, Both Directions, All D2D Pairs 
                  idx = dirIdx * numTransPerPair[srcDev][destDev]
                  tag = "cpu" + str(socket) + "_" + transTag[transIdx] + "_all_dev_dirs"
                  if (dirIdx == 0):
                     label = device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
                  else:
                     label = device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
                  add_scatter(blkSize, data[idx + prevIdx], color[colorIdx], marker[destDev * 2 + dirIdx], tag, label)
                  
                  # CASE 3: Each transfer type, Both Directions, All Sockets, All D2D Pairs
                  tag = transTag[transIdx] + "_all_cpu_dev_dirs"
                  if (dirIdx == 0):
                     label = "CPU " + str(socket) + " " + device[srcDev] + " to " + device[destDev] + " " + transLabel[transIdx]
                  else:
                     label = "CPU " + str(socket) + " " + device[destDev] + " to " + device[srcDev] + " " + transLabel[transIdx]
                  add_scatter(blkSize, data[idx + prevIdx], color[colorIdx2], marker[dirIdx * numSockets + socket], tag, label)
            colorIdx2 += 1
            prevIdx += numDirs
      # CASE 2: Each transfer type, Each Socket, Both Directions, All D2D Pairs 
      tag = "cpu" + str(socket) + "_" + transTag[transIdx] + "_all_dev_dirs"
      save_figure(tag, tag)
   
   # CASE 3: Each transfer type, Both Directions, All Sockets, All D2D Pairs
   tag = transTag[transIdx] + "_all_cpu_dev_dirs"
   save_figure(tag, tag)




