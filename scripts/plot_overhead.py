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
   numHostMemTypes = 3 # change to 5 if managed and mapped memory are added back

memTypes = ["Pageable", "Pinned", "Write-Combined", "Managed", "Mapped", "Device"]

devices = []
for idx in range(0, numDevices):
   devices.append(testParams[idx + 4]) 

print ("\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + ""
      "" + text.end + " given parameters:")
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "Device Count: " + str(numDevices)
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

ymax = max(np.amax(allocData), np.amax(freeData)) * 1.2
ymin = 0.1
xmax = int(blkSize[-1] * 1.2)
xmin = np.amin(blkSize)

#function for saving specific plot to file
def save_figure(tag, title):
   plt.figure(tag)
   plt.xscale('log')
   plt.yscale('log')
   #plt.ylim(ymax=ymax)
   plt.ylim(ymin=ymin)
   plt.xlim(xmax=xmax)
   plt.xlim(xmin=xmin)
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.5)

   #plt.title(title)
   plt.ylabel('Call Duration (us)')
   plt.xlabel('Freed Block Size (bytes)')
   plt.savefig("./overhead/" + tag + ".png", bbox_inches='tight', dpi=200)
   plt.clf()
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(label)
   plt.scatter(x, y, c = color, marker = mark, label = tag, linewidth=0.25, s=12) 
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
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
               node * (numHostMemTypes) + hostType

         # CASE 0
         allocLabel = "alloc_all_cpu_numa_mem_dev"
         freeLabel = "free_all_cpu_numa_mem_dev"
         allocTag = "CPU " + str(cpu) + " Node " + str(node) + " " + memTypes[hostType] 
         freeTag = "CPU " + str(cpu) + " Node " + str(node) + " " + memTypes[hostType] 
         colorIdx = (cpu * numNodes + node) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[cpu], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[cpu], freeTag, freeLabel)     
 
         # CASE 1
         allocLabel = "alloc_all_cpu_numa_mem_no_dev"
         freeLabel = "free_all_cpu_numa_mem_no_dev"
         allocTag = "CPU " + str(cpu) + " Node " + str(node) + " " + memTypes[hostType] 
         freeTag = "CPU " + str(cpu) + " Node " + str(node) + " " + memTypes[hostType]
         colorIdx = (cpu * numNodes * numHostMemTypes + node * numHostMemTypes + hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[hostType], freeTag, freeLabel)     

         # CASE 2
         allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
         allocTag = "Node " + str(node) + " " + memTypes[hostType]
         freeTag = "Node " + str(node) + " " + memTypes[hostType]
         colorIdx = (node * numHostMemTypes + hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[hostType], freeTag, freeLabel)     
 
         # CASE 3
         allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_no_dev"
         freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_no_dev"
         allocTag = "Node " + str(node) + " " + memTypes[hostType]
         freeTag = "Node " + str(node) + " " + memTypes[hostType]
         colorIdx = (node * numHostMemTypes + hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[hostType], freeTag, freeLabel)     

         # CASE 4: Each Socket, Each Node, All Devices, All Mem Types
         allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         allocTag = memTypes[hostType] 
         freeTag = memTypes[hostType]
         colorIdx = (hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[0], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[0], freeTag, freeLabel)     

      # CASE 4
      for dev in range(0, numDevices):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
               numNodes * (numHostMemTypes) + dev

         allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
         allocTag = devices[dev]
         freeTag = devices[dev]
         colorIdx = (numHostMemTypes + dev) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[dev + 1], allocTag, allocLabel)
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[dev + 1], freeTag, freeLabel) 

      # CASE 4 
      allocLabel = "alloc_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
      freeLabel = "free_cpu" + str(cpu) + "_numa" + str(node) + "_all_mem_dev"
      save_figure( allocLabel, allocLabel)
      save_figure( freeLabel, freeLabel) 
   
   for dev in range(0, numDevices):
      idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
            numNodes * (numHostMemTypes) + dev

      # CASE 2
      allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
      freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
      allocTag = devices[dev]
      freeTag = devices[dev]
      colorIdx = (numNodes * numHostMemTypes + dev) % len(color)
      add_scatter(blkSize, allocData[idx], color[colorIdx], marker[numNodes + dev], allocTag, allocLabel)  
      add_scatter(blkSize, freeData[idx], color[colorIdx], marker[numNodes + dev], freeTag, freeLabel)     
 
      # CASE 6
      allocLabel = "alloc_cpu" + str(cpu) + "_dev_only"
      freeLabel = "free_cpu" + str(cpu) + "_dev_only"
      allocTag = devices[dev]
      freeTag = devices[dev]
      colorIdx = (dev) % len(color)
      add_scatter(blkSize, allocData[idx], color[colorIdx], marker[0], allocTag, allocLabel)     
      add_scatter(blkSize, freeData[idx], color[colorIdx], marker[0], freeTag, freeLabel)     

   # CASE 6
   allocLabel = "alloc_cpu" + str(cpu) + "_dev_only"
   freeLabel = "free_cpu" + str(cpu) + "_dev_only"
   save_figure( allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel) 
   
   # CASE 3
   allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_no_dev"
   freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_no_dev"
   save_figure( allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel) 

   # CASE 2
   allocLabel = "alloc_cpu" + str(cpu) + "_all_numa_mem_dev"
   freeLabel = "free_cpu" + str(cpu) + "_all_numa_mem_dev"
   save_figure( allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel) 

for cpu in range(0, numSockets):
   for dev in range(0, numDevices):
      idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
            numNodes * (numHostMemTypes) + dev
    
      # CASE 0
      allocLabel = "alloc_all_cpu_numa_mem_dev"
      freeLabel = "free_all_cpu_numa_mem_dev"
      allocTag = "CPU " + str(cpu) + " " + devices[dev]
      freeTag = "CPU " + str(cpu) + " " + devices[dev]
      colorIdx = (numSockets * numNodes + numDevices * cpu + dev) % len(color)
      add_scatter(blkSize, allocData[idx], color[colorIdx], marker[cpu], allocTag, allocLabel)
      add_scatter(blkSize, freeData[idx], color[colorIdx], marker[cpu], freeTag, freeLabel)     

      # CASE 5
      allocLabel = "alloc_all_cpu_dev_only"
      freeLabel = "free_all_cpu_dev_only"
      allocTag = "CPU " + str(cpu) + " " + devices[dev]
      freeTag = "CPU " + str(cpu) + " " + devices[dev]
      colorIdx = (cpu * numDevices + dev) % len(color)
      add_scatter(blkSize, allocData[idx], color[colorIdx], marker[cpu], allocTag, allocLabel)
      add_scatter(blkSize, freeData[idx], color[colorIdx], marker[cpu], freeTag, freeLabel) 

# CASE 5
allocLabel = "alloc_all_cpu_dev_only"
freeLabel = "free_all_cpu_dev_only"
save_figure( allocLabel, allocLabel)
save_figure( freeLabel, freeLabel) 

# CASE 1
allocLabel = "alloc_all_cpu_numa_mem_no_dev"
freeLabel = "free_all_cpu_numa_mem_no_dev"
save_figure( allocLabel, allocLabel)
save_figure( freeLabel, freeLabel) 

# CASE 0 
allocLabel = "alloc_all_cpu_numa_mem_dev"
freeLabel = "free_all_cpu_numa_mem_dev"
save_figure( allocLabel, allocLabel)
save_figure( freeLabel, freeLabel) 

# CASE 7: Each Node, All Sockets, All Mem Types, All Devices
# CASE 8: Each Node, All Sockets, All Mem Types, No Devices
for node in range(0, numNodes):
   for hostType in range(0, numHostMemTypes):
      for cpu in range(0, numSockets):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
               node * (numHostMemTypes) + hostType
   
         # CASE 7
         allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
         freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
         allocTag = "CPU " + str(cpu) + " " + memTypes[hostType]
         freeTag = "CPU " + str(cpu) + " " + memTypes[hostType]
         colorIdx = (cpu * numHostMemTypes + hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[hostType], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[hostType], freeTag, freeLabel)     

         # CASE 8
         allocLabel = "alloc_node" + str(node) + "_all_cpu_mem_no_dev"
         freeLabel = "free_node" + str(node) + "_all_cpu_mem_no_dev"
         allocTag = "CPU " + str(cpu) + " " + memTypes[hostType]
         freeTag = "CPU " + str(cpu) + " " + memTypes[hostType]
         colorIdx = (cpu * numHostMemTypes + hostType) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[cpu], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[cpu], freeTag, freeLabel)     

   for cpu in range(0, numSockets):
      for dev in range(0, numDevices):
         idx = cpu * (numNodes * numHostMemTypes + numDevices) + \
               numNodes * (numHostMemTypes) + dev
  
         # CASE 7
         allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
         freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
         allocTag = "CPU " + str(cpu) + " " + devices[dev]
         freeTag = "CPU " + str(cpu) + " " + devices[dev]
         colorIdx = (numSockets * numHostMemTypes + dev) % len(color)
         add_scatter(blkSize, allocData[idx], color[colorIdx], marker[cpu], allocTag, allocLabel)     
         add_scatter(blkSize, freeData[idx], color[colorIdx], marker[cpu], freeTag, freeLabel)     

   # CASE 8
   allocLabel = "alloc_node" + str(node) + "_all_cpu_mem_no_dev"
   freeLabel = "free_node" + str(node) + "_all_cpu_mem_no_dev"
   save_figure( allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel) 

   # CASE 7
   allocLabel = "alloc_node" + str(node) + "_all_cpu_dev_mem"
   freeLabel = "free_node" + str(node) + "_all_cpu_dev_mem"
   save_figure( allocLabel, allocLabel)
   save_figure( freeLabel, freeLabel) 



