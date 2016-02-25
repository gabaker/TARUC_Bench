import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = list("brygcm")

if (len(sys.argv) != 5):
   print "Usage: python plot_overhead filename num_devices num_numa_nodes num_sockets"
   sys.exit() 

blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0))

numDevices = int(sys.argv[2])
numNumaNodes = int(sys.argv[3])
numSockets = int(sys.argv[4])
numCols = (numDevices + 2) * (numNumaNodes * numSockets)

alloc_data = []
dealloc_data = []

for idx in range(0, numCols):
   runLabel = ""
   
   nodeIdx = int(math.floor(idx / ((numDevices + 2) * numSockets)))
   socketIdx = int(math.floor((idx % ((numDevices + 2) * numNumaNodes)) / (numDevices + 2)))
   devIdx = idx % (numDevices + 2) 

   if (devIdx == 0):
      runLabel="cudaHostMalloc CPU:" + str(socketIdx) + " NODE:" + str(nodeIdx)
   elif(devIdx == 1):
      runLabel="malloc"
   else:
      runLabel="cudaMalloc Dev: " + str(devIdx - 2)

   plt.figure(1)
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx + 1))
   
   plt.plot(blkSize, x, c = colors[devIdx])
   plt.scatter(blkSize, x, c = colors[devIdx], label = runLabel)
   
   if (devIdx == 0):
      runLabel="cudaHostFree CPU:" + str(socketIdx) + " NODE:" + str(nodeIdx)
   elif(devIdx == 1):
      runLabel="free"
   else:
      runLabel="cudaFree Dev: " + str(devIdx - 2)

   plt.figure(2) 
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx + 2))
   plt.plot(blkSize, x, c = colors[devIdx])
   plt.scatter(blkSize, x, c = colors[devIdx], label = runLabel)

plt.figure(1)
plt.title('Memory Allocation Overhead')
plt.xlabel('Allocated Block Size (bytes)')
plt.ylabel('Call Duration (ms)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=8)
plt.ylim(ymin=0.0001)
plt.xlim(xmax=10000000000)
plt.savefig('overhead_alloc.png', bbox_inches='tight')

plt.figure(2)
plt.title('Memory Deallocation Overhead')
plt.ylabel('Call Duration (ms)')
plt.xlabel('Freed Block Size (bytes)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(ymin=0.0001)
plt.xlim(xmax=10000000000)
plt.legend(loc='upper left', fontsize=8)
plt.savefig('overhead_deallocation.png', bbox_inches='tight')

plt.show()
