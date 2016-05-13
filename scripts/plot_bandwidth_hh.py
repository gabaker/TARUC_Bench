import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

colors = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#800000', '#C0C0C0', '#800080', '#000000', '#00FFFF', '#A5522D']
marker=list("o^sDx*8.|-")

if (len(sys.argv) < 2):
   print "Usage: python script_name.py results_file.csv"
   sys.exit() 

if (os.path.isfile(sys.argv[1]) == False):
   print "Failed to open file: " + sys.argv[1]
   sys.exit()

#check if printing bandwidth or transfer time graphs
printBW = False
if (sys.argv[1].find("bw") != -1):
   printBW = True

results = open(sys.argv[1])
testParams = results.readline().strip().split(",");
numCols = len(results.readline().strip().split(","));

numSockets = int(testParams[0])
numNodes = int(testParams[1])
numPatterns = int(testParams[2])

testAllTrans = False
numTransTypes = 1
if (testParams[3] == "t"):
   testAllTrans = True
   numTransTypes = 5

useSockets = False
if (testParams[4] == "t"):
   useSockets = True
else:
   numSockets = 1

transLabel=["Both Pageable","Pinned Src","Pinned Dest","Both Pinned","Write-Combined Dest"]
transTag=["both_page", "pin_src","pin_dest","both_pin","wc_dest"]
patternLabel=["Repeated","Linear Inc","Linear Dec"]
patternTag=["repeat","linear_inc","linear_dec"]

print "\nPlotting H2H bandwidth results from file " + sys.argv[1] + " given parameters:"
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "# Transfer Types: " + str(numTransTypes)
print "# Access Pattern: " + str(numPatterns)
print "Test All Mem Types: " + str(testAllTrans)
print "Test All Sockets: " + str(useSockets)
print "Transfer Labels: " + str(transLabel)
print "Transfer Tags: " + str(transTag)
print "Patterns Label: " + str(patternLabel)
print "Patterns Tags: " + str(patternTag)

ylabel = ""
saveType = ""
xscale = 'log'
yscale = 'log'
if (printBW):
   ylabel = 'Copy Bandwidth (GB/S)'
   yscale = 'linear'
   #xscale = 'log'
   saveType = "hh_bw_"
else:
   ylabel = 'Transfer Time Per Block (us)'
   #xscale = 'log'
   #yscale = 'linear'
   saveType = "hh_tt_"

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))

#set print and save parameters depending on bw or tt type of graphs
data = []
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

xmax = int(blkSize[-1] * 2)
xmin = int(blkSize[0])
ymin = 0
#function for saving specific plot to file
def save_figure(figTag, title):
   plt.figure(figTag)
   plt.xscale(xscale)
   plt.yscale(yscale)
   plt.ylim(ymin=ymin)
   #plt.ylim(ymax=ymax)
   plt.xlim(xmin=xmin)
   plt.xlim(xmax=xmax)
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

   plt.title(title)
   plt.ylabel(ylabel)
   plt.xlabel('Copied Block Size (bytes)')
   plt.savefig("./bandwidth/hh/" + saveType + figTag + ".png", bbox_inches='tight')
   plt.clf()         
   return

def add_scatter(x, y, color, mark, tag, label):
   plt.figure(tag)
   plt.scatter(x, y, c = color, marker = mark, label = label) 
   return
#CASE -1: All
#CASE 0: Each socket, each src/dest pair, each pattern, all transfer types
#CASE 1: Each socket, each src/dest pair, each transfer type, all patterns
#CASE 2: Each socket, each pattern, each transfer type, all src/dest pairs
for socket in range(0, numSockets):
   for srcNode in range(0, numNodes):
      for destNode in range(0, numNodes):
         for patternIdx in range(0, numPatterns):
            for transIdx in range(0, numTransTypes):
               idx = socket * (numNodes * numNodes * numTransTypes * numPatterns) + \
                     srcNode * (numNodes * numTransTypes * numPatterns) + \
                     destNode * (numTransTypes * numPatterns) + \
                     transIdx * (numPatterns) + patternIdx
                        
               #CASE -1: All
               tag = "_all_cpu_src_dest_trans_types_patterns"
               label = "CPU: " + str(socket) + " Src: " + str(srcNode) + " Dest: " + str(destNode) + " " + transLabel[transIdx] + " " + patternLabel[patternIdx]
               add_scatter(blkSize, data[idx], colors[transIdx], marker[transIdx], tag, label)

               #CASE 0: Each socket, each src/dest pair, each pattern, all transfer types
               tag = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + patternTag[patternIdx] + "_all_tran_types"
               label = transLabel[transIdx] 
               add_scatter(blkSize, data[idx], colors[transIdx], marker[0], tag, label)
            
            #CASE 0: Each socket, each src/dest pair, each pattern, all transfer types
            tag = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + patternTag[patternIdx] + "_all_tran_types"
            save_figure(tag, tag)
         
         for transIdx in range(0, numTransTypes):
            for patternIdx in range(0, numPatterns):
               idx = socket * (numNodes * numNodes * numTransTypes * numPatterns) + \
                     srcNode * (numNodes * numTransTypes * numPatterns) + \
                     destNode * (numTransTypes * numPatterns) + \
                     transIdx * (numPatterns) + patternIdx
             
               #CASE 1: Each socket, each src/dest pair, each transfer type, all patterns
               tag = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_all_patterns"
               label = patternLabel[patternIdx] 
               add_scatter(blkSize, data[idx], colors[patternIdx], marker[patternIdx], tag, label)
            
            #CASE 1: Each socket, each src/dest pair, each transfer type, all patterns
            tag = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_all_patterns"
            save_figure(tag, tag)
 
   for patternIdx in range(0, numPatterns):
      for transIdx in range(0, numTransTypes):
         for srcNode in range(0, numNodes):
            for destNode in range(0, numNodes):
               idx = socket * (numNodes * numNodes * numTransTypes * numPatterns) + \
                     srcNode * (numNodes * numTransTypes * numPatterns) + \
                     destNode * (numTransTypes * numPatterns) + \
                     transIdx * (numPatterns) + patternIdx
               
               #CASE 2: Each socket, each pattern, each transfer type, all src/dest pairs
               tag = "cpu" + str(socket) + "_" + patternTag[patternIdx] + "_" + transTag[transIdx] + "_all_src_dest"
               label = "Src Node: " + str(srcNode) + " Dest Node: " + str(destNode)
               add_scatter(blkSize, data[idx], colors[srcNode * numNodes + destNode], marker[srcNode], tag, label)
         
         #CASE 2: Each socket, each pattern, each transfer type, all src/dest pairs
         tag = "cpu" + str(socket) + "_" + patternTag[patternIdx] + "_" + transTag[transIdx] + "_all_src_dest"
         save_figure(tag, tag)

#CASE -1: All
tag = "_all_cpu_src_dest_trans_types_patterns"
save_figure(tag, tag)


#CASE 3: Each transfer type, each src/dest pair, each pattern, all sockets
#CASE 4: Each transfer type, each src/dest pair, all patterns, all sockets
#CASE 5: Each Transfer type, each pattern, all sockets, all src/dest pairs  
if (numSockets > 1):
   for transIdx in range(0, numTransTypes):
      for srcNode in range(0, numNodes):
         for destNode in range(0, numNodes):
            for patternIdx in range(0, numPatterns):
               for socket in range(0, numSockets):
                  idx = socket * (numNodes * numNodes * numTransTypes * numPatterns) + \
                        srcNode * (numNodes * numTransTypes * numPatterns) + \
                        destNode * (numTransTypes * numPatterns) + \
                        transIdx * (numPatterns) + patternIdx

                  #CASE 3: Each transfer type, each src/dest pair, each pattern, all sockets
                  tag = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_" + patternTag[patternIdx]
                  label = "CPU: " + str(socket)
                  add_scatter(blkSize, data[idx], colors[socket], marker[socket], tag, label)
            
                  #CASE 4: Each transfer type, each src/dest pair, all patterns, all sockets
                  tag = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_all_patterns"
                  label = "CPU: " + str(socket) + " Pattern: " + patternLabel[patternIdx]
                  add_scatter(blkSize, data[idx], colors[socket * numPatterns + patternIdx], marker[socket], tag, label)

               #CASE 3: Each transfer type, each src/dest pair, each pattern, all sockets
               tag = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_" + patternTag[patternIdx]
               save_figure(tag, tag)
            
            #CASE 4: Each transfer type, each src/dest pair, all patterns, all sockets
            tag = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_" + transTag[transIdx] + "_all_patterns"
            save_figure(tag, tag)
         
      for patternIdx in range(0, numPatterns):
         for socket in range(0, numSockets):
            for srcNode in range(0, numNodes):
               for destNode in range(0, numNodes):
                  idx = socket * (numNodes * numNodes * numTransTypes * numPatterns) + \
                        srcNode * (numNodes * numTransTypes * numPatterns) + \
                        destNode * (numTransTypes * numPatterns) + \
                        transIdx * (numPatterns) + patternIdx
 
                  #CASE 5: Each Transfer type, each pattern, all sockets, all src/dest pairs  
                  tag = "all_cpu_src_dest_" + transTag[transIdx] + "_" + patternTag[patternIdx]
                  label = "CPU: " + str(socket) + " Src Node: " + str(srcNode) + " Dest Node: " + str(destNode)
                  add_scatter(blkSize, data[idx], colors[srcNode * numNodes + destNode], marker[socket], tag, label)

         #CASE 5: Each Transfer type, each pattern, all sockets, all src/dest pairs  
         tag = "all_cpu_src_dest_" + transTag[transIdx] + "_" + patternTag[patternIdx]
         save_figure(tag, tag)


