import numpy as np
import matplotlib.pyplot as plt
import sys
import os


if (len(sys.argv) != 3):
   print "Usage: python plot_overhead filename num_devices"
   sys.exit() 
#os.path.isfile(fname)

numDevices = int(sys.argv[2])

colors = list("brygcm")

blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0))

for idx in range(0, numDevices):
   runLabel = ""

   if (idx == 0):
      runLabel="cudaHostMalloc"
   elif(idx == 1):
      runLabel="malloc"
   else:
      runLabel="cudaMalloc " + str(idx - 2)

   plt.figure(1)
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx * 2 + 1))
   plt.scatter(blkSize, x, c = colors[idx], label = runLabel)
   
   if (idx == 0):
      runLabel="cudaHostFree"
   elif(idx == 1):
      runLabel="free"
   else:
      runLabel="cudaFree " + str(idx - 2)

   plt.figure(2) 
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx * 2 + 2))
   plt.scatter(blkSize, x, c = colors[idx], label = runLabel)


plt.figure(1)
plt.title('Memory Allocation Overhead')
plt.xlabel('Allocated Block Size (bytes)')
plt.ylabel('Call Duration (ms)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=8)
plt.ylim(ymin=0.01)
plt.xlim(xmax=10000000000)
plt.savefig('overhead_alloc.png', bbox_inches='tight')

plt.figure(2)
plt.title('Memory Deallocation Overhead')
plt.ylabel('Call Duration (ms)')
plt.xlabel('Freed Block Size (bytes)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(ymin=0.01)
plt.xlim(xmax=10000000000)
plt.legend(loc='upper left', fontsize=8)
plt.savefig('overhead_deallocation.png', bbox_inches='tight')

plt.show()

