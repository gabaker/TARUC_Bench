import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#800000', '#C0C0C0', '#800080', '#000000', '#00FFFF', '#A5522D']
marker=list("o^sDx")

if (len(sys.argv) < 2):
   print "Usage: python plot_overhead.py results_file.csv"
   sys.exit() 

results = open(sys.argv[1])
testParams = results.readline().strip().split(",");
numSockets = int(testParams[0])
numNodes = int(testParams[1])
numDevices = int(testParams[2])

numHostMemTypes = 1
testAllMemTypes = False
if (testParams[3] == "t"):
   testAllMemTypes = True
   numHostMemTypes = 3

memAllocTypes = ["Page Host","Pinned Host", "Write-Combined", "Device"]
memFreeTypes = ["Page Free","Pinned Free", "Write-Combined Free", "Device Free"]

devices = []
for idx in range(0, numDevices):
   devices.append(testParams[idx + 4]) 

print "\nPlotting memory overhead results from file " + sys.argv[1] + " given parameters:"
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "Num Devices: " + str(numDevices)
print "Test All Mem Types: " + str(testAllMemTypes)
print "Devices: " + str(devices)

# read each column into the correct location, in order
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))
allocData = []
freeData = []

numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   if (idx % 2):
      allocData.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))
   else:
      freeData.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

ymax = max(np.amax(allocData), np.amax(freeData))
ymax = math.pow(10, math.log10(ymax)) * 2
ymin = 0.100
xmin = 0.0
xmax = int(blkSize[-1] * 2)
#function for saving specific plot to file
def save_figure( figureNum, title, saveName ):
   plt.figure(figureNum)
   plt.xscale('log')
   plt.yscale('log')
   plt.ylim(ymax=ymax)
   plt.ylim(ymin=ymin)
   plt.xlim(xmax=xmax)
   plt.xlim(xmin=xmin)
   plt.legend(loc='upper left', fontsize=8)

   plt.title(title)
   plt.ylabel('Call Duration (ms)')
   plt.xlabel('Freed Block Size (bytes)')

   plt.savefig("./results/" + saveName + ".png", bbox_inches='tight')
   plt.clf()
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(label)
   plt.scatter(x, y, c = color, marker = mark, label = tag) 
   return

# CASE 0: All Sockets, All Nodes, All Mem Types, All Devices
# CASE 1: All Sockets, All Nodes, All Mem Types, No Devices
# CASE 2: Each Socket, All Nodes, All Mem Types, All Devices 
# CASE 3: Each Socket, All Nodes, All Mem Types, No Devices 
# CASE 4: Each Socket, Each Node, All Devices, All Mem Types
# CASE 5: All Sockets, All Devices, No Host
# CASE 6: Each Socket, All Devices, No Host
for cpu in range(0, numSockets):
   for node in range(0, numNodes):
      for hostType in range(0, numHostMemTypes):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + (node * numHostMemTypes) + hostType
         yAlloc = allocData[idx] 
         yFree = freeData[idx] 

         # CASE 0
         allocLabel = "alloc_all_cpu_numa_mem_dev"
         freeLabel = "free_all_cpu_numa_mem_dev"
         allocTag = memAllocTypes[hostType] + " CPU " + str(cpu) + " NUMA " + str(node)
         freeTag = memFreeTypes[hostType] + " CPU " + str(cpu) + " NUMA " + str(node)
         add_scatter(blkSize, yAlloc, colors[cpu * numNodes + node], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[cpu * numNodes + node], marker[hostType], freeTag, freeLabel)     
 
         # CASE 1
         allocLabel = "alloc_all_cpu_numa_mem_no_dev"
         freeLabel = "free_all_cpu_numa_mem_no_dev"
         allocTag = memAllocTypes[hostType] + " CPU " + str(cpu) + " NUMA " + str(node)
         freeTag = memFreeTypes[hostType] + " CPU " + str(cpu) + " NUMA " + str(node)
         add_scatter(blkSize, yAlloc, colors[cpu * numNodes + node], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[cpu * numNodes + node], marker[hostType], freeTag, freeLabel)     

         # CASE 2
         allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
         allocTag = memAllocTypes[hostType] + " NUMA " + str(node)
         freeTag = memFreeTypes[hostType] + " NUMA " + str(node)
         add_scatter(blkSize, yAlloc, colors[cpu * numNodes + node], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[cpu * numNodes + node], marker[hostType], freeTag, freeLabel)     
 
         # CASE 3
         allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_no_dev"
         freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_no_dev"
         allocTag = memAllocTypes[hostType] + " NUMA " + str(node)
         freeTag = memFreeTypes[hostType] + " NUMA " + str(node)
         add_scatter(blkSize, yAlloc, colors[cpu * numNodes + node], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[cpu * numNodes + node], marker[hostType], freeTag, freeLabel)     

         # CASE 4
         allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         allocTag = memAllocTypes[hostType] 
         freeTag = memFreeTypes[hostType]
         add_scatter(blkSize, yAlloc, colors[node * numHostMemTypes + hostType], marker[0], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[node * numHostMemTypes + hostType], marker[0], freeTag, freeLabel)     

      # CASE 4
      for dev in range(0, numDevices):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + (numNodes * numHostMemTypes) + dev
         yAlloc = allocData[idx] 
         yFree = freeData[idx] 

         allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         allocTag = memAllocTypes[3] + " " + devices[dev]
         freeTag = memFreeTypes[3]+ " " + devices[dev]
         add_scatter(blkSize, yAlloc, colors[dev], marker[dev + 1], allocTag, allocLabel)
         add_scatter(blkSize, yFree, colors[dev], marker[dev + 1], freeTag, freeLabel) 

      # CASE 4 
      allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
      freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
      save_figure( allocLabel, allocLabel, allocLabel)
      save_figure( freeLabel, freeLabel, freeLabel) 
   
   for dev in range(0, numDevices):
      idx = cpu * (numNodes * numHostMemTypes + numDevices) + (numNodes * numHostMemTypes) + dev
      yAlloc = allocData[idx] 
      yFree = freeData[idx] 

      # CASE 2
      allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
      freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
      allocTag = memAllocTypes[3] + " " + devices[dev]
      freeTag = memFreeTypes[3] + " " + devices[dev]
      add_scatter(blkSize, yAlloc, colors[numHostMemTypes * numNodes + dev], marker[dev], allocTag, allocLabel)  
      add_scatter(blkSize, yFree, colors[numHostMemTypes * numNodes + dev], marker[dev], freeTag, freeLabel)     
      
      # CASE 6
      allocLabel = "alloc_cpu" + str(cpu) + " _dev_only"
      freeLabel = "free_cpu" + str(cpu) + " _dev_only"
      allocTag = memAllocTypes[2] + " " + devices[dev]
      freeTag = memFreeTypes[2] + " " + devices[dev]
      add_scatter(blkSize, yAlloc, colors[dev], marker[0], allocTag, allocLabel)     
      add_scatter(blkSize, yFree, colors[dev], marker[0], freeTag, freeLabel)     

   # CASE 6
   allocLabel = "alloc_cpu" + str(cpu) + " _dev_only"
   freeLabel = "free_cpu" + str(cpu) + " _dev_only"
   save_figure( allocLabel, allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel, freeLabel) 
   
   # CASE 3
   allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_no_dev"
   freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_no_dev"
   save_figure( allocLabel, allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel, freeLabel) 

   # CASE 2
   allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
   freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
   save_figure( allocLabel, allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel, freeLabel) 

for cpu in range(0, numSockets):
   for dev in range(0, numDevices):
      idx = cpu * (numNodes * numHostMemTypes + numDevices) + (numNodes * numHostMemTypes) + dev
      yAlloc = allocData[idx] 
      yFree = freeData[idx] 
     
      # CASE 0
      allocLabel = "alloc_all_cpu_numa_mem_dev"
      freeLabel = "free_all_cpu_numa_mem_dev"
      allocTag = memAllocTypes[3] + " CPU " + str(cpu) + " " + devices[dev]
      freeTag = memFreeTypes[3] + " CPU " + str(cpu) + " " + devices[dev]
      add_scatter(blkSize, yAlloc, colors[numNodes * numSockets + dev], marker[cpu], allocTag, allocLabel)
      add_scatter(blkSize, yFree, colors[numNodes * numSockets + dev], marker[cpu], freeTag, freeLabel)     

      # CASE 5
      allocLabel = "alloc_all_cpu_dev_only"
      freeLabel = "free_all_cpu_dev_only"
      allocTag = memAllocTypes[3] + " CPU " + str(cpu) + " " + devices[dev]
      freeTag = memFreeTypes[3] + " CPU " + str(cpu) + " " + devices[dev]
      add_scatter(blkSize, yAlloc, colors[cpu * numSockets + dev], marker[cpu], allocTag, allocLabel)
      add_scatter(blkSize, yFree, colors[cpu * numSockets + dev], marker[cpu], freeTag, freeLabel) 

# CASE 5
allocLabel = "alloc_all_cpu_dev_only"
freeLabel = "free_all_cpu_dev_only"
save_figure( allocLabel, allocLabel, allocLabel)
save_figure( freeLabel, freeLabel, freeLabel) 

# CASE 1
allocLabel = "alloc_all_cpu_numa_mem_no_dev"
freeLabel = "free_all_cpu_numa_mem_no_dev"
save_figure( allocLabel, allocLabel, allocLabel)
save_figure( freeLabel, freeLabel, freeLabel) 

# CASE 0 
allocLabel = "alloc_all_cpu_numa_mem_dev"
freeLabel = "free_all_cpu_numa_mem_dev"
save_figure( allocLabel, allocLabel, allocLabel)
save_figure( freeLabel, freeLabel, freeLabel) 

# CASE 7: Each Node, All Sockets, All Mem Types, All Devices
# CASE 8: Each Node, All Sockets, All Mem Types, No Devices
for node in range(0, numNodes):
   for hostType in range(0, numHostMemTypes):
      for cpu in range(0, numSockets):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + (node * numHostMemTypes) + hostType
         yAlloc = allocData[idx] 
         yFree = freeData[idx] 
    
         # CASE 7
         allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
         freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
         allocTag = memAllocTypes[hostType] + " CPU " + str(cpu)
         freeTag = memFreeTypes[hostType] + " CPU " + str(cpu)
         add_scatter(blkSize, yAlloc, colors[hostType * numSockets + cpu], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[hostType * numSockets + cpu], marker[hostType], freeTag, freeLabel)     

         # CASE 8
         allocLabel = "alloc_node" + str(node) + "_all_cpu_mem_no_dev"
         freeLabel = "free_node" + str(node) + "_all_cpu_mem_no_dev"
         allocTag = memAllocTypes[hostType] + " CPU " + str(cpu)
         freeTag = memFreeTypes[hostType] + " CPU " + str(cpu)
         add_scatter(blkSize, yAlloc, colors[hostType], marker[cpu], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[hostType], marker[cpu], freeTag, freeLabel)     

   for cpu in range(0, numSockets):
      for dev in range(0, numDevices):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + (numNodes * numHostMemTypes) + dev
         yAlloc = allocData[idx] 
         yFree = freeData[idx] 
   
         # CASE 7
         allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
         freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
         allocTag = memAllocTypes[2] + " CPU " + str(cpu) + " " + devices[dev]
         freeTag = memFreeTypes[2] + " CPU " + str(cpu) + " " + devices[dev]
         add_scatter(blkSize, yAlloc, colors[numSockets * numHostMemTypes + dev], marker[cpu], allocTag, allocLabel)     
         add_scatter(blkSize, yFree, colors[numSockets * numHostMemTypes + dev], marker[cpu], freeTag, freeLabel)     

   # CASE 8
   allocLabel = "alloc_node" + str(node) + "_all_cpu_mem_no_dev"
   freeLabel = "free_node" + str(node) + "_all_cpu_mem_no_dev"
   save_figure( allocLabel, allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel, freeLabel) 

   # CASE 7
   allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
   freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
   save_figure( allocLabel, allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel, freeLabel) 































 
