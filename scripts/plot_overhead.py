import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = list("brygcm")

if (len(sys.argv) != 5):
   print "Usage: python plot_overhead filename num_devices num_numa_nodes num_sockets"
   sys.exit() 

numNumaNodes = int(sys.argv[2])
numSockets = int(sys.argv[3])
numDevices = int(sys.argv[4])
numCols = (numDevices + 2) * (numNumaNodes * numSockets)

#function for saving specific plot to file
def save_figure( figureNum, title, saveName ):
   plt.figure(figureNum)
   plt.xscale('log')
   plt.yscale('log')
   plt.ylim(ymin=0.0001)
   plt.xlim(xmax=10000000000)
   plt.legend(loc='upper left', fontsize=8)

   plt.title(title)
   plt.ylabel('Call Duration (ms)')
   plt.xlabel('Freed Block Size (bytes)')

   plt.savefig("./results/" + saveName + ".png", bbox_inches='tight')
   return

# function for clearing all plots between graphing runs
def clear_plot_range( begin, end):
   for plotIdx in range(begin, end + 1):
      plt.figure(plotIdx)
      plt.clf()
   return

# read each column into the correct location, in order
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0))
alloc_data = []
free_data = []

for idx in range(0, numCols):
   alloc_data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(2 * idx + 1)))
   free_data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(2 * idx + 2)))

# CASE 0: all NUMA nodes, all CPUs, all devices
# CASE 1: each NUMA node, all CPUs, all devices
# CASE 2: each NUMA node, each CPU, all devices
# CASE 3: each NUMA node, each CPU, each device
# CASE 4: all NUMA nodes, all CPUs, no devices
for node in range(0, numNumaNodes):
   for cpu in range(0, numSockets):
      cpuIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node)
      label = "CPU" + str(cpu) + " NUMA" + str(node)
            
      cpu_alloc_y0 = alloc_data[cpuIdx]
      cpu_alloc_y1 = alloc_data[cpuIdx + 1]
      cpu_free_y0 = free_data[cpuIdx]
      cpu_free_y1 = free_data[cpuIdx + 1]
      
      #CASE 0:
      plt.figure("alloc" + str(0))
      plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
      plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 
      plt.figure("free" + str(0))
      plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
      plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

      #CASE 1:
      plt.figure("alloc" + str(1))
      plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
      plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 
      plt.figure("free" + str(1))
      plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
      plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

      #CASE 2:
      plt.figure("alloc" + str(2))
      plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
      plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 
      plt.figure("free" + str(2))
      plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
      plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

      #CASE 4:   
      plt.figure("alloc" + str(4))
      plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
      plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 
      plt.figure("free" + str(4))
      plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
      plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

      for dev in range(0, numDevices):
         devIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node) + (2 + dev) 
         dev_alloc_y = alloc_data[devIdx]
         dev_free_y = free_data[devIdx]
         devAllocLabel = "cudaMalloc " + label + " DEV" + str(dev)
         devFreeLabel = "cudaFree " + label + " DEV" + str(dev)
         
         #CASE 0:
         plt.figure("alloc" + str(0)) 
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.figure("free" + str(0)) 
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 

         #CASE 1:
         plt.figure("alloc" + str(1))
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.figure("free" + str(1))
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
 
         #CASE 2: 
         plt.figure("alloc" + str(2))
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.figure("free" + str(2))
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
     
         #CASE 3: 
         plt.figure("alloc" + str(3))
         plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
         plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 

         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         save_figure("alloc" + str(3), "", "alloc_numa" + str(node) + "_cpu" + str(cpu) +"_dev" + str(dev))
         plt.clf()
         
         plt.figure("free" + str(3))
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
         save_figure("free" + str(3), "", "free_numa" + str(node) + "_cpu" + str(cpu) +"_dev" + str(dev))
         plt.clf()
     
      #CASE 2: 
      save_figure("alloc" + str(2), "", "alloc_numa" + str(node) + "_cpu" + str(cpu) +"_all_dev")
      plt.clf()
      save_figure("free" + str(2), "", "free_numa" + str(node) + "_cpu" + str(cpu) +"_all_dev")
      plt.clf()         

   #CASE 1: 
   save_figure("alloc" + str(1), "", "alloc_numa" + str(node) + "_all_cpu_dev")
   plt.clf()         
   save_figure("free" + str(1), "", "free_numa" + str(node) + "_all_cpu_dev")
   plt.clf()         

#CASE 0
save_figure("alloc" + str(0), "", "alloc_all_numa_cpu_dev")
save_figure("free" + str(0), "", "free_all_numa_cpu_dev")

#CASE 4
save_figure("alloc" + str(4), "", "alloc_all_numa_cpu_nodev")
save_figure("free" + str(4), "", "free_all_numa_cpu_nodev")

# CASE 5: all CPUs, each NUMA node, each DEV
for node in range(0, numNumaNodes):
   for dev in range(0, numDevices):
      for cpu in range(0, numSockets):
         label = "CPU" + str(cpu) + " NUMA" + str(node)   
         
         cpuIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node)
         devIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node) + (2 + dev) 

         cpu_alloc_y0 = alloc_data[cpuIdx]
         cpu_alloc_y1 = alloc_data[cpuIdx + 1]
         cpu_free_y0 = free_data[cpuIdx]
         cpu_free_y1 = free_data[cpuIdx + 1]
         dev_alloc_y = alloc_data[devIdx]
         dev_free_y = free_data[devIdx]

         devAllocLabel = "cudaMalloc " + label + " DEV" + str(dev)
         devFreeLabel = "cudaFree " + label + " DEV" + str(dev)
         
         #CASE 5:
         plt.figure("alloc" + str(5))
         plt.scatter(blkSize, cpu_alloc_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
         plt.scatter(blkSize, cpu_alloc_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 

         if (cpu == 0):
            plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 

         plt.figure("free" + str(5))
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 

         if (cpu == 0):
            plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
 
      #CASE 5:
      save_figure("alloc" + str(5), "", "alloc_numa" + str(node) + "_all_cpu_dev" + str(dev))
      save_figure("free" + str(5), "", "free_numa" + str(node) + "_all_cpu_dev" + str(dev)) 

# CASE 6: each CPU, all NUMA nodes, each DEV
# CASE 7: each CPU, all NUMA nodes, all DEVs
label = ""
for cpu in range(0, numSockets):     
   for dev in range(0, numDevices):     
      for node in range(0, numNumaNodes):
         cpuIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node)
         cpu_alloc_y0 = alloc_data[cpuIdx]
         cpu_alloc_y1 = alloc_data[cpuIdx + 1]
         cpu_free_y0 = free_data[cpuIdx]
         cpu_free_y1 = free_data[cpuIdx + 1]

         label = "CPU" + str(cpu) + " NUMA" + str(node)
         devAllocLabel = "cudaMalloc " + label + " DEV" + str(dev)
         devFreeLabel = "cudaFree " + label + " DEV" + str(dev)
         
         devIdx = ((numDevices + 2) * cpu) + ((numDevices + 2) * numSockets * node) + (2 + dev) 
         dev_alloc_y = alloc_data[devIdx]
         dev_free_y = free_data[devIdx]

         #CASE 6
         plt.figure("alloc" + str(6))
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.figure("free" + str(6))
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 
 
         #CASE 7
         plt.figure("alloc" + str(7))
         plt.scatter(blkSize, dev_alloc_y, c = colors[dev], label = devAllocLabel) 
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaMallocHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "malloc " + label) 

         plt.figure("free" + str(7))
         plt.scatter(blkSize, dev_free_y, c = colors[dev], label = devFreeLabel) 
         plt.scatter(blkSize, cpu_free_y0, c = colors[node * 2 + cpu], label = "cudaFreeHost " + label) 
         plt.scatter(blkSize, cpu_free_y1, c = colors[node * 2 + cpu + 1], label = "free " + label) 
   
      #CASE 6
      save_figure("alloc" + str(6), "", "alloc_all_numa_cpu" + str(cpu) + "_dev" + str(dev))
      plt.clf()
      save_figure("free" + str(6), "", "free_all_numa_cpu" + str(cpu) +"_dev" + str(dev))
      plt.clf()         
   
   #CASE 7
   save_figure("alloc" + str(7), "", "alloc_all_numa_cpu" + str(cpu) +"_all_dev")
   plt.clf()
   save_figure("free" + str(7), "", "free_all_numa_cpu" + str(cpu) +"_all_dev")
   plt.clf()         

''' 
   plt.figure(2) 
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx + 2))
   plt.plot(blkSize, x, c = colors[devIdx])
   plt.scatter(blkSize, x, c = colors[devIdx], label = runLabel)

'''
