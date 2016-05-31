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

# read in results file header values
results = open(sys.argv[1])
testParams = results.readline().strip().split(",");

numSockets = int(testParams[0])
numNodes = int(testParams[1])
numMemTypes = int(testParams[2])
memTag = ["pin","page"]
memLabel = ["Pinned","Pageable"]

print "\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + text.end + " given parameters:"
print "Socket Count: " + str(numSockets)
print "NUMA Node Count: " + str(numNodes)
print "Host Mem Types: " + str(numMemTypes)
print "Memory Labels: " + str(memLabel)
print "Memory Tags: " + str(memTag)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

avgData = data

#set print and save parameters depending on bw or tt type of graphs
xmax = int(blkSize[-1] * 1.2)
xmin = blkSize[0]
ymin = 0

#for idx in range(1,numCols):

#function for saving specific plot to file
def save_figure(tag, title):
   plt.figure(tag)
   plt.xscale('log')
   plt.yscale('log')
   plt.xlim(xmin=xmin)
   plt.xlim(xmax=xmax)
   plt.ylim(ymin=ymin)
   #plt.ylim(ymax=ymax)
   
   #plt.title(title)
   plt.ylabel("Total Access Time (us)")
   plt.xlabel("Number of Doubles Accessed")
   plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=10, labelspacing=0.50)
   plt.savefig("random_access/" + tag + ".png", bbox_inches='tight', dpi=150, markersize=20)
   plt.clf()
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(tag)
   plt.scatter(x, y, c = color, marker = mark, label = label, linewidth=0.25, s=10) 
   return

# CASE 0: All 
# CASE 1: Each Memory Type, All socket, All src nodes, All nodes
# CASE 2: Each Memory Type, Each Socket, All src/dest nodes
for memIdx in range(0, numMemTypes): 
   for socket in range(0, numSockets):
      for srcNode in range(0, numNodes):
         for destNode in range(0, numNodes):
            idx = memIdx * (numSockets * numNodes * numNodes) + \
                  socket * (numNodes * numNodes) + \
                  srcNode * (numNodes) + destNode

            # CASE 0: All 
            tag = "all_mem_cpu_nodes"
            label = memLabel[memIdx] + " CPU: " + str(socket) + " Src: " + str(srcNode) + " Dest: " + str(destNode)
            colorIdx = (memIdx * numSockets * numNodes * numNodes + socket * numNodes * numNodes + srcNode * numNodes + destNode) % len(color)
            add_scatter(blkSize, data[idx], color[colorIdx], marker[memIdx * numSockets + socket], tag, label)

            # CASE 1: Each Memory Type, All socket, All src nodes, All nodes
            tag = memTag[memIdx] + "_all_cpu_nodes"
            label = "CPU: " + str(socket) + " Src: " + str(srcNode) + " Dest: " + str(destNode)
            colorIdx = (socket * numNodes * numNodes + srcNode * numNodes + destNode) % len(color) 
            add_scatter(blkSize, data[idx], color[colorIdx], marker[colorIdx], tag, label)

            # CASE 2: Each Memory Type, Each Socket, All src/dest nodes
            tag = memTag[memIdx] + "_cpu" + str(socket) + "_all_nodes"
            label = " Src: " + str(srcNode) + " Dest: " + str(destNode) 
            colorIdx = (srcNode * numNodes + destNode) % len(color)
            add_scatter(blkSize, data[idx], color[colorIdx], marker[colorIdx], tag, label)

      # CASE 2: Each Memory Type, Each Socket, All src/dest nodes
      tag = memTag[memIdx] + "_cpu" + str(socket) + "_all_nodes"
      save_figure(tag, tag)

   # CASE 1: Each Memory Type, All socket, All src nodes, All nodes
   tag = memTag[memIdx] + "_all_cpu_nodes"
   save_figure(tag, tag)

# CASE 0: All 
tag = "all_mem_cpu_nodes"
save_figure(tag, tag)

# CASE 3: Each Socket, All Memory Types. All src/dest nodes
for socket in range(0, numSockets):
   for memIdx in range(0, numMemTypes): 
      for srcNode in range(0, numNodes):
         for destNode in range(0, numNodes):
            idx = memIdx * (numSockets * numNodes * numNodes) + \
                  socket * (numNodes * numNodes) + \
                  srcNode * (numNodes) + destNode

            # CASE 3: Each Socket, All Memory Types. All src/dest nodes
            tag = "cpu" + str(socket) + "_all_mem_nodes"
            label = memLabel[memIdx] + " Src: " + str(srcNode) + " Dest: " + str(destNode) 
            colorIdx = (memIdx * numNodes * numNodes + srcNode * numNodes + destNode) % len(color) 
            add_scatter(blkSize, data[idx], color[colorIdx], marker[srcNode * numNodes + destNode], tag, label)

   # CASE 3: Each Socket, All Memory Types. All src/dest nodes
   tag = "cpu" + str(socket) + "_all_mem_nodes"
   save_figure(tag, tag)

# CASE 4: Each src/dest node, All Sockets, All Memory Types
for srcNode in range(0, numNodes):
   for destNode in range(0, numNodes):
      for memIdx in range(0, numMemTypes): 
         for socket in range(0, numSockets):
            idx = memIdx * (numSockets * numNodes * numNodes) + \
                  socket * (numNodes * numNodes) + \
                  srcNode * (numNodes) + destNode

            # CASE 4: Each src/dest node, All Sockets, All Memory Types
            tag = "src" + str(srcNode) + "_dest" + str(destNode) + "_all_mem_cpu"
            label = memLabel[memIdx] + " CPU: " + str(socket) 
            colorIdx = (memIdx * numSockets + socket) % len(color) 
            add_scatter(blkSize, data[idx], color[colorIdx], marker[colorIdx], tag, label)

      # CASE 4: Each src/dest node, All Sockets, All Memory Types
      tag = "src" + str(srcNode) + "_dest" + str(destNode) + "_all_mem_cpu"
      save_figure(tag, tag)






