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
#transLabel = ["D2D","P2P"]

devices = []
for idx in range(0, numDevices):
   devices.append(testParams[idx + 4]) 

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
print "Devices: " + str(devices)
print numTransPerPair

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
   #plt.clf()
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(tag)
   plt.scatter(x, y, c = color, marker = mark, label = label, linewidth = 0.25, s=12) 
   return

numDirs = 2
prevIdx = 0
label = ""
tag = ""

# CASE 0: Each Socket, Each D2D Pair, All Transfer Types, Both Directions
# CASE 1: Each Socket, Each D2D Pair, Each Transfer Type, Both Directions
# CASE 2: Each D2D Pair, Each Transfer Type, Both Directions, All Sockets
# CASE 3: Each D2D Pair, All Transfer Types, Both Directions, All Sockets
# CASE 4: All Intra-node Transfers

for socket in range(0, numSockets):
   for srcDev in range(0, numDevices):
      for destDev in range(srcDev, numDevices):     
         if (srcDev == destDev):
            numDirs = 1
         else:
            numDirs = 2
         
         for transIdx in range(0, numTransPerPair[srcDev][destDev]):
            for dirIdx in range(0, numDirs):
               idx =  prevIdx + transIdx * numDirs + dirIdx

               if (srcDev != destDev):
                  # CASE 0: Each Socket, Each D2D Pair, All Transfer Types, Both Directions
                  tag = "cpu" + str(socket) + "_dev" + str(srcDev) + "_" + str(destDev) + "_all_trans_types_dirs"
                  if (dirIdx == 0):
                     label = "Src: " + devices[srcDev] + "\nDest: " + devices[destDev] + "\n" + transLabel[transIdx]
                  else:
                     label = "Src: " + devices[destDev] + "\nDest: " + devices[srcDev] + "\n" + transLabel[transIdx]
                  add_scatter(blkSize, data[idx], color[transIdx * numDirs + dirIdx], marker[transIdx * numDirs + dirIdx], tag, label) 
                  
                  # CASE 1: Each Socket, Each D2D Pair, Each Transfer Type, Both Directions
                  tag = "cpu" + str(socket) + "_dev_" + str(srcDev) + "_" + str(destDev) + "_" + transTag[transIdx] + "_all_dirs"
                  if (dirIdx == 0):
                     label = "Src: " + devices[srcDev] + "\nDest: " + devices[destDev]
                  else:
                     label = "Src: " + devices[destDev] + "\nDest: " + devices[srcDev]
                  add_scatter(blkSize, data[idx], color[dirIdx], marker[dirIdx], tag, label) 

                  # CASE 2: Each D2D Pair, Each Transfer Type, Both Directions, All Sockets
                  tag = "dev_" + str(srcDev) + "_" + str(destDev) + "_" + transTag[transIdx] + "_all_dirs_cpus"
                  if (dirIdx == 0):
                     label = "CPU " + str(socket) + "\nSrc: " + devices[srcDev] + "\nDest: " + devices[destDev]
                  else:
                     label = "CPU " + str(socket) + "\nSrc: " + devices[destDev] + "\nDest: " + devices[srcDev]
                  add_scatter(blkSize, data[idx], color[socket * numDirs + dirIdx], marker[socket * numDirs + dirIdx], tag, label) 

                  # CASE 3: Each D2D Pair, All Transfer Types, Both Directions, All Sockets
                  tag = "dev_" + str(srcDev) + "_" + str(destDev) + "_all_trans_types_dirs_cpus"
                  if (dirIdx == 0):
                     label = "CPU " + str(socket) + "\nSrc: " + devices[srcDev] + "\nDest: " + devices[destDev] + "\n" + transLabel[transIdx]
                  else:
                     label = "CPU " + str(socket) + "\nSrc: " + devices[destDev] + "\nDest: " + devices[srcDev] + "\n" + transLabel[transIdx]
                  add_scatter(blkSize, data[idx], color[transIdx * numDirs * numSockets + dirIdx * numSockets + socket], marker[dirIdx * numSockets + socket], tag, label) 
    
               if (srcDev == destDev):
                  # CASE 4: All Intra-gpu Transfers
                  tag = "cpu" + str(socket) + "_" + "all_dev_intra_gpu_trans"
                  label = "" + devices[srcDev] 
                  add_scatter(blkSize, data[idx], color[srcDev], marker[srcDev], tag, label) 
       
         prevIdx += numDirs * numTransPerPair[srcDev][destDev]

# Save graphs for case 2 + 3
for socket in range(0, numSockets):
   # CASE 4: All Intra-gpu Transfers
   tag = "cpu" + str(socket) + "_" + "all_dev_intra_gpu_trans"
   save_figure(tag, tag)

   for srcDev in range(0, numDevices):
      for destDev in range(srcDev, numDevices):     

         if (srcDev != destDev):
            # CASE 0: Each Socket, Each D2D Pair, All Transfer Types, Both Directions
            tag = "cpu" + str(socket) + "_dev" + str(srcDev) + "_" + str(destDev) + "_all_trans_types_dirs"
            save_figure(tag, tag)

            # CASE 3: Each D2D Pair, All Transfer Types, Both Directions, All Sockets
            tag = "dev_" + str(srcDev) + "_" + str(destDev) + "_all_trans_types_dirs_cpus"
            save_figure(tag, tag)

         for transIdx in range(0, numTransPerPair[srcDev][destDev]):
            
            if (srcDev != destDev): 
               # CASE 1: Each Socket, Each D2D Pair, Each Transfer Type, Both Directions
               tag = "cpu" + str(socket) + "_dev_" + str(srcDev) + "_" + str(destDev) + "_" + transTag[transIdx] + "_all_dirs"
               save_figure(tag, tag)

               # CASE 2: Each D2D Pair, Each Transfer Type, Both Directions, All Sockets
               tag = "dev_" + str(srcDev) + "_" + str(destDev) + "_" + transTag[transIdx] + "_all_dirs_cpus"
               save_figure(tag, tag)














