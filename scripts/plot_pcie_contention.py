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

results = open(sys.argv[1])
testParams = results.readline().strip().split(",")

numCPUs = int(testParams[0])
numThreads = int(testParams[1])
numGPUs = int(testParams[2])
numDirs = int(testParams[3])
numCPUTests = int(testParams[4])

devices = []
for idx in range(0, numGPUs):
   devices.append(testParams[idx + 5]) 

dirTag = ["h2d","d2h","both_dirs"]
dirLabel = ["Host-to-Device","Device-to-Host","Bidirectional"]
dirLabelShort = ["H2D","D2H","Both"]

data = np.genfromtxt(str(sys.argv[1]), delimiter=",", skip_header=1, usecols=0)
ymax = np.max(data)
threads = np.arange(1, numThreads + 1)

print ("\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + ""
      "" + text.end + " given parameters:")
print "Socket Count: " + str(numCPUs)
print "Thread Count: " + str(numThreads)
print "Device Count: " + str(numGPUs)
print "Directions: " + str(numDirs)
print "Socket Tests: " + str(numCPUTests)
print "Devices: " + str(devices)
print "Direction Labels: " + str(dirLabel)
print "Direction Tags: " + str(dirTag)

def save_figure(tag, title, numTicks, subfolder):
   plt.figure(tag)
   plt.title(title)
   plt.ylim(ymax=ymax)
   plt.ylabel("Transfer Bandwidth (GB/s)")
   plt.xlabel('Number of Concurrent Threads')
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.50)
   plt.xticks(np.arange(1, numTicks + 1)) 
   plt.savefig("./contention/pcie/" + subfolder + title + ".png", bbox_inches='tight', dpi=150)
   plt.clf()
   return

def add_chart(x, y, color, tag, label, w):
   plt.figure(tag)
   plt.bar(x, y, width=w, color=color, label=label, linewidth=0.25) 
   return

tag = ""
label = ""
dataIdx = 0

# Single Device, Multiple Host Thread Contention
# CASE 0: Each Device, Each Direction, Each Socket 
# CASE 1: Each Device, Each Direction, All Sockets
# CASE 2: Each Device, Each Socket, All Directions 
for devIdx in range(0, numGPUs):
   for dirIdx in range(0, numDirs):
      for cpuIdx in range(0, numCPUTests):
         y = data[dataIdx:dataIdx + numThreads] 
         
         # CASE 1: Each Device, Each Direction, Each Socket 
         label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
         tag = "node" + str(cpuIdx) + "_dev" + str(devIdx) + "_" + dirTag[dirIdx]
         if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
            label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
            tag = "all_node_test_dev" + str(devIdx) + "_" + dirTag[dirIdx]
         
         numBars = 1
         offset = 0.96
         shift = 0.0  - 0.48
         add_chart(threads + shift, y, color[dirIdx], tag, label, offset)
         save_figure(tag, tag, numThreads, "single_gpu/")
        
         # CASE 1: Each Device, Each Direction, All Sockets
         tag = "dev" + str(devIdx) + "_" + dirTag[dirIdx] + "_all_nodes"
         label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
         if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
            label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]

         numBars = numCPUTests
         offset = 0.96 / numBars
         shift = cpuIdx * offset - 0.48
         add_chart(threads + shift, y, color[cpuIdx], tag, label, offset)

         # CASE 2: Each Device, Each Socket, All Directions 
         label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
         tag = "node" + str(cpuIdx) + "_dev" + str(devIdx) + "_all_copy_dirs"
         if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
            label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
            tag = "nodes_test_dev" + str(devIdx) + "_all_copy_dirs" 
         
         numBars = numDirs
         offset = 0.96 / numBars
         shift = dirIdx * offset - 0.48
         add_chart(threads + shift, y, color[dirIdx], tag, label, offset)
 
         # The results are essentially one column; increment by num threads 
         # after data is added to charts  to get to next data set
         dataIdx += numThreads

      # CASE 1: Each Device, Each Direction, All Sockets
      tag = "dev" + str(devIdx) + "_" + dirTag[dirIdx] + "_all_nodes"
      save_figure(tag, tag, numThreads, "single_gpu/")

   # Save bar chart for case 2
   for cpuIdx in range(0, numCPUTests):
      # CASE 2: Each Device, Each Socket, All Directions 
      tag = "node" + str(cpuIdx) + "_dev" + str(devIdx) + "_all_copy_dirs"
      if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
         tag = "nodes_test_dev" + str(devIdx) + "_all_copy_dirs"
      save_figure(tag, tag, numThreads, "single_gpu/")

if (numGPUs > 1):
   
   numGPUPairs = math.factorial(numGPUs - 1) 
   
   # GPU Pair Contention
   # CASE 3: Each Node, Each Device Pair, Each Direction 
   # CASE 4: Each Node, Each Device Pair, All Direction
   # CASE 5: Each Node, Each Direction, All Device Pairs
   # CASE 5.5: Each Direction, Each Device Pair, All Nodes 
   for cpuIdx in range(0, numCPUTests): 
      pairIdx = 0
      for devIdx1 in range(0, numGPUs):
         for devIdx2 in range(devIdx1 + 1, numGPUs):
            for dirIdx in range(0, numDirs):
               y = data[dataIdx:dataIdx + numThreads] 
               
               # CASE 1: Each Device, Each Direction, Each Socket 
               numBars = 1
               offset = 0.96
               shift = 0.0  - 0.48
               
               label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
               tag = "node" + str(cpuIdx) + "_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_" + dirTag[dirIdx]
               if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
                  label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
                  tag = "nodes_test_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_" + dirTag[dirIdx]
               
               add_chart(threads + shift, y, color[cpuIdx], tag, label, offset)
               save_figure(tag, tag, numThreads, "gpu_pair/")

               # CASE 4: Each Node, Each Device Pair, All Direction
               numBars = numDirs
               offset = 0.96 / numBars
               shift = dirIdx * offset  - 0.48
               
               label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
               tag = "node" + str(cpuIdx) + "_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_all_copy_dirs"
               if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
                  label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
                  tag = "nodes_test_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_all_copy_dirs"
               
               add_chart(threads + shift, y, color[dirIdx], tag, label, offset)

               # CASE 5: Each Node, Each Direction, All Device Pairs
               numBars = numGPUPairs
               offset = 0.96 / numBars
               shift = pairIdx * offset - 0.48
               
               label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
               tag = "node" + str(cpuIdx) + "_" + dirTag[dirIdx] + "_all_dev_pairs"
               if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
                  label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
                  tag = "nodes_test_" + dirTag[dirIdx] + "_all_dev_pairs"
               
               add_chart(threads + shift, y, color[pairIdx % len(color)], tag, label, offset)

               # CASE 5.5: Each Direction, Each Device Pair, All Nodes 
               numBars = numCPUTests
               offset = 0.96 / numBars
               shift = cpuIdx * offset  - 0.48
               
               label = "Node " + str(cpuIdx) + "\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
               tag = "dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_" + dirTag[dirIdx] + "_all_nodes" 
               if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
                  label = "All Nodes\n" + devices[devIdx] + "\n" + dirLabel[dirIdx]
               
               add_chart(threads + shift, y, color[cpuIdx], tag, label, offset)

               # incrememt the location in data array by the number of threads
               dataIdx += numThreads

            # move onto the next device pair
            pairIdx += 1

            # CASE 4: Each Node, Each Device Pair, All Direction
            tag = "node" + str(cpuIdx) + "_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_all_copy_dirs"
            if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
               tag = "nodes_test_dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_all_copy_dirs"
            save_figure(tag, tag, numThreads, "gpu_pair/")

      # Save bar charts for case 5
      for dirIdx in range(0, numDirs):
         
         # CASE 5: Each Node, Each Direction, All Device Pairs
         tag = "node" + str(cpuIdx) + "_" + dirTag[dirIdx] + "_all_dev_pairs"
         if (((cpuIdx + 1) == numCPUs) & (numCPUs > 1)):
            tag = "nodes_test_" + dirTag[dirIdx] + "_all_dev_pairs"
         save_figure(tag, tag, numThreads, "gpu_pair/")

   for devIdx1 in range(0, numGPUs):
      for devIdx2 in range(devIdx1 + 1, numGPUs):
         for dirIdx in range(0, numDirs):
         
            # CASE 5.5: Each Direction, Each Device Pair, All Nodes 
            tag = "dev_pair_" + str(devIdx1) + "_" + str(devIdx2) + "_" + dirTag[dirIdx] + "_all_nodes" 
            save_figure(tag, tag, numThreads, "gpu_pair/")

   # Single Host Multiple Device Contention
   # CASE 6: All Sockets, All Directions
   # CASE 7: Each Socket, Each Direction
   # CASE 8: Each Socket, All Directions
   # CASE 9: Each Direction, All Sockets
   for cpuIdx in range(0, numCPUs):
      for dirIdx in range(0, numDirs):
         y = data[dataIdx:dataIdx + numThreads] 

         # CASE 6: All
         numBars = numCPUs * numDirs
         offset = 0.9 / numBars
         shift = (cpuIdx * numDirs + dirIdx) * offset - 0.45
         tag = "all_nodes_copy_dirs"
         label = "Node " + str(cpuIdx) + "\n" + dirLabel[dirIdx]
         add_chart(threads + shift, y, color[cpuIdx * numDirs + dirIdx], tag, label, offset)

         # CASE 7: Each Socket, Each Direction
         numBars = 1
         offset = 0.9
         shift = 0.0  - 0.45
         label = "Node " + str(cpuIdx) + "\n" + dirLabel[dirIdx]
         tag = "node" + str(cpuIdx) + "_" + dirTag[dirIdx] 
         add_chart(threads + shift, y, color[dirIdx], tag, label, offset)
         save_figure(tag, tag, numThreads, "single_node/")

         # CASE 8: Each Socket, All Directions
         numBars = numDirs
         offset = 0.9 / numBars
         shift = dirIdx * offset - 0.45
         label = "Node " + str(cpuIdx) + "\n" + dirLabel[dirIdx]
         tag = "node" + str(cpuIdx) + "_all_copy_dirs"
         label = dirLabel[dirIdx]
         add_chart(threads + shift, y, color[dirIdx], tag, label, offset)

         # CASE 9: Each Direction, All Sockets
         numBars = numCPUs
         offset = 0.9 / numBars
         shift = cpuIdx * offset - 0.45
         tag = dirTag[dirIdx] + "_all_nodes"
         label = "Node " + str(cpuIdx) + " \n" + dirLabel[dirIdx]
         add_chart(threads + shift, y, color[cpuIdx], tag, label, offset)
    
         # The results are essentially one column; increment by num threads 
         # after data is added to charts  to get to next data set
         dataIdx += numThreads
    
      # CASE 6: Each Socket, All Directions
      tag = "node" + str(cpuIdx) + "_all_copy_dirs"
      save_figure(tag, tag, numThreads, "single_node/")

   # CASE 6: All
   tag = "all_nodes_copy_dirs"
   save_figure(tag, tag, numThreads, "single_node/")

   # Save bar charts for case 12
   for dirIdx in range(0, numDirs):
      # CASE 9: Each Direction, All Sockets
      tag = dirTag[dirIdx] + "_all_nodes"
      save_figure(tag, tag, numThreads, "single_node/")

