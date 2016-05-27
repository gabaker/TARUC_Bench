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

# read in results file header values
results = open(sys.argv[1])
testParams = results.readline().strip().split(",");

# Parse parameters and read in data
numThreads = int(testParams[0])
numDirs = int(testParams[1])
numOps = int(testParams[2])
threads = np.arange(1, numThreads + 1)
data = np.genfromtxt(str(sys.argv[1]), delimiter=",", skip_header=1, usecols=0)
ymax = np.max(data) * 1.1

dirTag = ["uni_dir","bi_dir"]
dirLabel = ["Unidirectional", "Bidirectional"]
opTag = ["memcpy","copy", "triad"]
opLabel = ["Memcpy()","Manual Copy","Triad (Copy, Scale, Add)"]

print ("\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + ""
      "" + text.end + " given parameters:")
print "Max Threads: " + str(numThreads)
print "Direction Count: " + str(numDirs)
print "Number of Operations: " + str(numOps)
print "Operation Tags: " + str(opTag)
print "Operation Labels: " + str(opLabel)
print "Transfer Direction Tags: " + str(dirTag)
print "Transfer Direction Labels: " + str(dirLabel)

def save_figure(tag, title, numTicks):
   plt.figure(tag)
   plt.title(title)
   plt.ylim(ymax=ymax)

   plt.ylabel("Transfer Bandwidth (GB/s)")
   plt.xlabel('Number of Concurrent Threads')
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.75)
   plt.xticks(np.arange(1, numTicks + 1)) 
   plt.savefig("./contention/qpi/" + title + ".png", bbox_inches='tight', dpi=200)
   plt.clf()
   return

def add_chart(x, y, color, tag, label, w):
   plt.figure(tag)
   plt.bar(x, y, width=w, color=color, label=label, linewidth=0.25) 
   return

# CASE 0: All
# CASE 1: Each direction, Each Op
# CASE 2: Each direction, All Ops
# CASE 3: Each Op, All directions
dataIdx = 0
for dirIdx in range(0, numDirs):
   for opIdx in range(0, numOps):
      y = data[dataIdx:dataIdx + numThreads] 
      x = threads[0:numThreads] 
       
      # CASE 0: All
      numBars = numDirs * numOps
      barOffset = 0.9 / numBars
      barShift = (dirIdx * numOps + opIdx) * barOffset - 0.45
      tag = "all_dirs_mem_ops"
      label = dirLabel[dirIdx] + "\n" + opLabel[opIdx]
      add_chart(x + barShift, y, color[dirIdx * numOps + opIdx], tag, label, barOffset)

      # CASE 1: Each direction, Each Op
      numBars = 1
      barOffset = 0.9
      barShift = 0.0  - 0.45
      tag = dirTag[dirIdx] + "_" + opTag[opIdx]
      label = ""
      add_chart(x + barShift, y, color[opIdx], tag, label, barOffset)
      save_figure(tag, tag, numThreads)

      # CASE 2: Each direction, All Ops
      numBars = numOps
      barOffset = 0.9 / numBars
      barShift = opIdx * barOffset - 0.45
      tag = dirTag[dirIdx] + "_all_mem_ops"
      label = opLabel[opIdx]
      add_chart(x + barShift, y, color[opIdx], tag, label, barOffset)

      # CASE 3: Each Op, All directions
      numBars = numDirs
      barOffset = 0.9 / numBars
      barShift = dirIdx * barOffset - 0.45
      tag = "" + opTag[opIdx] + "_all_nodes"
      label = dirLabel[dirIdx]
      add_chart(x + barShift, y, color[dirIdx], tag, label, barOffset)
 
      # The results are essentially one column; increment by num threads 
      # after data is added to charts  to get to next data set
      dataIdx += numThreads
 
   # CASE 2: Each direction, All Ops
   tag = dirTag[dirIdx] + "_all_mem_ops"
   save_figure(tag, tag, numThreads)

# CASE 0: All
tag = "all_dirs_mem_ops"
save_figure(tag, tag, numThreads)

# Save bar charts for case 3
for opIdx in range(0, numOps):
   # CASE 3: Each Op, All directions
   tag = "" + opTag[opIdx] + "_all_nodes"
   save_figure(tag, tag, numThreads)





