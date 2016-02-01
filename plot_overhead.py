import numpy as np
import matplotlib.pyplot as plt
import sys
import os


if (len(sys.argv) != 3):
   print "Usage: python plot_overhead filename num_devices"
   sys.exit() 
#os.path.isfile(fname)

numDevices = int(sys.argv[2])

colors = list("bgrmcy")

blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0))

for idx in range(0, numDevices):
   devName = "Device - " + str(idx)

   plt.figure(1)
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx + 1))
   plt.scatter(blkSize, x, c = colors[idx], label = devName)
   
   plt.figure(2) 
   x = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx + 2))
   plt.scatter(blkSize, x, c = colors[idx], label = devName)


plt.figure(1)
plt.title('Device Memory Allocation Overhead')
plt.xlabel('Allocated Block Size (in Bytes)')
plt.ylabel('Call Duration')

plt.xscale('log')
plt.yscale('log')
plt.savefig('device_alloc.png', bbox_inches='tight')

plt.figure(2)
plt.title('Device Free Memory Overhead')
plt.ylabel('Call Duration')
plt.xlabel('Allocated Block Size (in Bytes)')
plt.xscale('log')
plt.yscale('log')
plt.savefig('device_free.png', bbox_inches='tight')
plt.show()


