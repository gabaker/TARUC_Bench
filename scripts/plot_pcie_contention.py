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
marker=list("o^sDx*8.|h15p+_")

if (len(sys.argv) < 2):
   print "Usage: python script_name.py results_file.csv"
   sys.exit() 

if (os.path.isfile(sys.argv[1]) == False):
   print "Failed to open file: " + sys.argv[1]
   sys.exit()

results = open(sys.argv[1])
testParams = []
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));

numNodes = int(testParams[0][0])
numGPUs = int(testParams[0][1])
blkSize = int(testParams[0][2])

devices = []
for idx in range(0, numGPUs):
   devices.append(testParams[0][idx + 3]) 

devSingleNode = int(testParams[1][0])
devAllNodes = int(testParams[1][1])
pairThreads = int(testParams[2][0])
hostThreads = int(testParams[3][0])
data = np.genfromtxt(str(sys.argv[1]), delimiter=",", skip_header=4, usecols=0)
maxBW = np.max(data)

print "\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + text.end + " given parameters:"
print "Socket Count: " + str(numNodes)
print "Device Count: " + str(numGPUs)
print "Devices: " + str(devices)
print "Thread Counts For Subtests: "
print "Single Device, One Node: " + str(devSingleNode)
print "Single Device, All Nodes: " + str(devAllNodes)
print "Device Pair: " + str(pairThreads)
print "Host, All Devices: " + str(hostThreads)

dirTag = ["h2d","d2h","both"]

def save_figure(tag, title):
   plt.figure(tag)
   plt.xlim(xmin=0)
   plt.ylim(ymin=0)
   plt.autoscale(tight=True)
   plt.ylim(ymax=maxBW)
   
   plt.title(title)
   plt.ylabel("Transfer Bandwidth (GB/s)")
   plt.xlabel('Number of Consecutive Threads')
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
   plt.savefig("./contention/pcie/" + title + ".png", bbox_inches='tight')
   plt.clf()
   return

def add_chart(x, y, color, tag, label, w):
   plt.figure(tag)
   #plt.xticks(x,x + w)
   plt.bar(x, y, width=w, color = color, label = label, align='center') 
   return

y = []
x = []
xTicks = np.arange(1, max((devSingleNode, devAllNodes, pairThreads, hostThreads)) + 1)
numDirs = 3
numSocketTests = numNodes
if (numNodes > 1):
   numSocketTests += 1
idx0 = 0
offset = 0
tag=""

# Single Device, Multiple Host Thread Contention
for devIdx in range(0, numGPUs):
   for dirIdx in range(0, numDirs):
      for nodeIdx in range(0, numSocketTests):
         if (nodeIdx == numNodes):
            offset = devAllNodes
         else:
            offset = devSingleNode
 
         x = xTicks[0:offset] 
         y = data[idx0:idx0+offset] 
         
         # CASE 0: Each Device, Each Direction, Each Socket 
         if (nodeIdx == numNodes):
            tag = "0_dev" + str(devIdx) + "_both_nodes_dir_"+ dirTag[dirIdx]
         else:
            tag = "0_dev" + str(devIdx) + "_node" + str(nodeIdx) + "_dir_"+ dirTag[dirIdx]
         
         label = ""
         add_chart(x, y, color[0], tag, label, 0.9) 
         save_figure(tag, tag)
         
         # CASE 1: Each Device, Each Direction, All Sockets
         if (numNodes > 1):
            tag = "0_dev" + str(devIdx) + "_dir_"+ dirTag[dirIdx] + "_all_nodes"
            
            label = "Node " + str(nodeIdx)
            if (nodeIdx == numNodes):
               label = "All Nodes"
            shift = nodeIdx * .3 - .3
            add_chart(x + shift, y, color[nodeIdx], tag, label, .3) 

         # CASE 2: Each Device, Each Socket, All Directions 
         tag = "0_dev" + str(devIdx) + "_node_"+ str(nodeIdx) + "_all_dirs"
          
         label = "Node " + str(nodeIdx) + " to " + devices[devIdx]
         if (dirIdx == 1):
            label = devices[devIdx] + " to Node " + str(nodeIdx)
         elif (dirIdx == 2):
            label = "Bidirectional"
         shift = dirIdx * .3 - .3 
         add_chart(x + shift, y, color[dirIdx], tag, label, .3) 
         
         idx0 += offset

      # CASE 1: Each Device, Each Direction, All Sockets
      if (numNodes > 1):
         tag = "0_dev" + str(devIdx) + "_dir_"+ dirTag[dirIdx] + "_all_nodes"
         save_figure(tag, tag)

   for nodeIdx in range(0, numSocketTests):        
      # CASE 2: Each Device, Each Socket, All Directions 
      tag = "0_dev" + str(devIdx) + "_node_"+ str(nodeIdx) + "_all_dirs"
      save_figure(tag, tag)

xTicks = np.arange(numNodes, pairThreads, numNodes)

# GPU Pair Contention
# CASE 3: Each Node, Each Device Pair, Each Direction 
# CASE 4: Each Node, Each Device Pair, All Direction
for dev1 in range(0, numGPUs):
   for dev2 in range(dev1 + 1, numGPUs):
      for dirIdx in range(0, len(dirTag)):      
         offset = pairThreads
         x = xTicks[0:offset] 
         y = data[idx0:idx0+offset] 
 
         offset = offset / numNodes        
         
# CASE 5: Each Direction, All Device Pairs
# CASE 6: All Device Pairs, All Direction
for dirIdx in range(0, len(dirTag)):      
   for dev1 in range(0, numGPUs):
      for dev2 in range(dev1 + 1, numGPUs):
         test = 1

# Single Host Multiple Device Contention
# CASE 7: Each Socket, Each Direction
for nodeIdx in range(0, numNodes):
   for dirIdx in range(0, len(dirTag)):
      test = 1

# CASE 10: All Sockets, All Directions
# CASE 9: Each Direction, All Sockets
if (numNodes > 1):
   for dirIdx in range(0, len(dirTag)):
      for nodeIdx in range(0, numNodes):
         test = 1








