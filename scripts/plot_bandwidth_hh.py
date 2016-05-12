import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

#colors = list("brygcmk")
colors = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#800000', '#C0C0C0', '#800080', '#000000', '#00FFFF', '#A5522D']
marker=list("o^sDx")
#"", 
if (len(sys.argv) < 2):
   print "Usage: python script_name.py results_file.csv[ device_info.out]"
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

numSockets = int(testParams[0])
numNodes = int(testParams[1])

usePinnedMem = False
numMemTypes = 1
if (testParams[2] == "t"):
   usePinnedMem = True
   numMemTypes = 6

startSocket = 0
useSockets = False
if (testParams[3] == "t"):
   useSockets = True
else:
   startSocket = numSockets
   numSockets = 1

memTypeNames={"Pageable","Pinned","Write-Combined"}
patternNames=[]
numPatterns = len(testParams) - 4
for idx in range(0, numPatterns):
   patternNames.append(testParams[idx + 4])

print "\nPlotting H2H bandwidth results from file " + sys.argv[1] + " given parameters:"
print "Socket Count: " + str(numSockets)
print "Node Count: " + str(numNodes)
print "Use Pinned Memory: " + str(usePinnedMem)
print "Use All Sockets: " + str(useSockets)
print "Patterns: " + str(patternNames)

# read transfer block size for each ranged step
blkSize = np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(0), skip_header=(1))

#set print and save parameters depending on bw or tt type of graphs
plot_xmax = int(blkSize[-1] * 1.5)
plot_ymin = 0
saveType = ""
xscale = 'log'
yscale = 'log'
ylabel = ''
 
if (printBW):
   ylabel = 'Copy Bandwidth (GB/S)'
   yscale = 'linear'
   xscale = 'log'
   saveType = "hh_bw_"
   plot_ymin = 0.001
else:
   ylabel = 'Transfer Time Per Block (ms)'
   xscale = 'log'
   yscale = 'linear'
   saveType = "hh_tt_"
   plot_ymin = 0

data = []
numCols = len(results.readline().strip().split(","));
for idx in range(1, numCols):
   data.append(np.genfromtxt (str(sys.argv[1]), delimiter=",", usecols=(idx), skip_header=(1)))

#function for saving specific plot to file
def save_figure(figTag, title, saveName):
   plt.figure(figTag)
   plt.xscale(xscale)
   plt.yscale(yscale)
   plt.ylim(ymin=plot_ymin)
   plt.xlim(xmax=plot_xmax)
   plt.legend(loc='upper left', fontsize=8)

   plt.title(title)
   plt.ylabel(ylabel)
   plt.xlabel('Block Size (bytes)')

   plt.savefig("./results/" + saveType + saveName + ".png", bbox_inches='tight')
   return

#CASE 0: Each socket, all mem combinations, each src/dest pair, one pattern
#CASE 1: Each socket, each mem combination, all src/dest pairs, one pattern
#CASE 2: Each socket, each mem combination, each src/dest pair, all patterns
for socket in range(startSocket, startSocket + numSockets):
   for srcNode in range(0, numNodes):
      for destNode in range(0, numNodes):
         y_idx =  socket * numSockets * numNodes * numMemTypes * numPatterns + srcNode * numNodes * numMemTypes * numPatterns + destNode * numMemTypes * numPatterns
         
         #CASE 0: Each socket, all mem combinations, each src/dest pair, one pattern
         label = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_mem_types"
         plt.figure(label)
         plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[0], label = "Both Page") 
         if (usePinnedMem): 
            plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[1], label = "Pinned Src") 
            plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[2], label = "Pinned Dest") 
            plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[3], label = "Both Pinned") 

         save_figure(label, "H2H: CPU " + str(socket) + " Src " + str(srcNode)+ " Dest " + str(destNode) + ", All Mem Types", label)
         plt.clf()         

         label = "cpu" + str(socket) + "_all_src_dest"
         #CASE 1: Both Pageable, all src/dest pairs, repeated pattern, each socket
         plt.figure(label + "_both_page")
         plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[srcNode * numNodes + destNode], label = "Src " + str(srcNode) + " Pinned Dest " + str(destNode)) 
         
         if (usePinnedMem): 
            #CASE 1: Source Pinned, all src/dest pairs, repeated pattern, each socket
            plt.figure(label + "_pinned_src")
            plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[srcNode * numNodes + destNode], label = "Pinned Src " + str(srcNode) + " Pinned Dest " + str(destNode)) 

            #CASE 1: Destination Pinned, all src/dest pairs, repeated pattern, each socket
            plt.figure(label + "_pinned_dest")
            plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[srcNode * numNodes + destNode], label = "Src " + str(srcNode) + " Pinned Dest " + str(destNode)) 

            #CASE 1: Both Pinned, all src/dest pairs, repeated pattern, each socket
            plt.figure(label + "_both_pinned")
            plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[srcNode * numNodes + destNode], label = "Src " + str(srcNode) + " Dest " + str(destNode)) 

         for pattern in range(0, numPatterns):
            pattIdx = pattern + y_idx
            label = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_patterns"
            #CASE 2: Both Pageable, one src/dest pair, all patterns, each socket
            plt.figure(label + "_both_page")
            plt.scatter(blkSize, data[pattIdx + 0 * numPatterns], c = colors[pattern], label = patternNames[pattern]) 
           
            if (usePinnedMem): 
               #CASE 2: Source Pinned, one src/dest pairs, all patterns, each socket
               plt.figure(label + "_pinned_src")
               plt.scatter(blkSize, data[pattIdx + 1 * numPatterns], c = colors[pattern], label = patternNames[pattern]) 

               #CASE 2: Destination Pinned, one src/dest pair, all pattern, each socket
               plt.figure(label + "_pinned_dest")
               plt.scatter(blkSize, data[pattIdx + 2 * numPatterns], c = colors[pattern], label = patternNames[pattern]) 

               #CASE 2: Both Pinned, one src/dest pairs, all pattern, each socket
               plt.figure(label + "_both_pinned")
               plt.scatter(blkSize, data[pattIdx + 3 * numPatterns], c = colors[pattern], label = patternNames[pattern]) 

         label = "cpu" + str(socket) + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_patterns"
         #CASE 2: Both Pageable, one src/dest pair, all patterns, each socket
         save_figure(label + "_both_page", "H2H: All Patterns, Both Pageable, CPU " + str(socket) + " Src "+ str(srcNode) + " Dest " + str(destNode), label + "_both_page")
         plt.clf()         
           
         if (usePinnedMem): 
            #CASE 2: Source Pinned, one src/dest pair, all patterns, each socket
            save_figure(label + "_pinned_src", "H2H: All Patterns, Src Pinned, CPU " + str(socket) + " Src "+ str(srcNode) + " Dest " + str(destNode), label + "_pinned_src")
            plt.clf()         

            #CASE 2: Destination Pinned, one src/dest pair, all pattern, each socket
            save_figure(label + "_pinned_dest", "H2H: All Patterns, Dest Pinned, CPU " + str(socket) + " Src "+ str(srcNode) + " Dest " + str(destNode), label + "_pinned_dest")
            plt.clf()         

            #CASE 2: Both Pinned, one src/dest pair, all pattern, each socket
            save_figure(label + "_both_pinned", "H2H: All Patterns, Both Pinned, CPU " + str(socket) + " Src "+ str(srcNode) + " Dest " + str(destNode), label + "_both_pinned")
            plt.clf()         
      
   #CASE 1: Both Pageable, all src/dest pairs, repeated pattern, each socket
   label = "cpu" + str(socket) + "_all_src_dest"
   save_figure(label + "_both_page", "H2H: Socket " + str(socket) + ", Both Pageable, All Host Pairs", label + "_both_page")
   plt.clf()         
  
   if (usePinnedMem): 
      plt.figure(label + "_pinned_src")
      save_figure(label + "_pinned_src", "H2H: Socket " + str(socket) + ", Pinned Src, All Host Pairs", label + "_pinned_src")
      plt.clf()         

      #CASE 1: Destination Pinned, all src/dest pairs, repeated pattern, each socket
      save_figure(label + "_pinned_dest", "H2H: Socket " + str(socket) + ", Pinned Dest, All Host Pairs", label + "_pinned_dest")
      plt.clf()         

      #CASE 1: Both Pinned, all src/dest pairs, repeated pattern, each socket
      save_figure(label + "_both_pinned", "H2H: Socket " + str(socket) + ", Both Pinned, All Host Pairs", label + "_both_pinned")
      plt.clf()         
 
if (numSockets > 1):
   #CASE 3: All sockets, each mem combination, each src/dest pair, one pattern
   #CASE 4: All sockets, each mem combination, each src/dest pair, all patterns
   for srcNode in range(0, numNodes):
      for destNode in range(0, numNodes):
         for socket in range(startSocket, startSocket + numSockets):
            y_idx =  socket * numSockets * numNodes * numMemTypes * numPatterns + srcNode * numNodes * numMemTypes * numPatterns + destNode * numMemTypes * numPatterns
            label = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode)
            #CASE 3: All sockets, both pageable, one pattern and one src/dest pair
            plt.figure(label + "_both_page")
            plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[socket], label = "CPU " + str(socket) + " Src: " + str(srcNode) + "Dest: " + str(destNode)) 
            
            if (usePinnedMem): 
               #CASE 3: All sockets, pinned source, one pattern and one src/dest pair
               plt.figure(label + "_pinned_src")
               plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[socket], label = "CPU " + str(socket) + " Src: " + str(srcNode) + "Dest: " + str(destNode)) 

               #CASE 3: All sockets, pinned destination, one pattern and one src/dest pair
               plt.figure(label + "_pinned_dest")
               plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[socket], label = "CPU " + str(socket) + " Src: " + str(srcNode) + "Dest: " + str(destNode)) 

               #CASE 3: All sockets, both pinned, one pattern and one src/dest pair
               plt.figure(label + "_both_pinned")
               plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[socket], label = "CPU " + str(socket) + " Src: " + str(srcNode) + "Dest: " + str(destNode)) 

            for pattern in range(0, numPatterns):
               y_idx =  socket * numSockets * numNodes * numMemTypes * numPatterns + srcNode * numNodes * numMemTypes * numPatterns + destNode * numMemTypes * numPatterns + pattern
               #CASE 4: All sockets, all patterns, each src/dest pair, both pageable
               label = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_patterns"
               plt.figure(label + "_both_page")
               plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[pattern], marker = marker[socket] ,label = "CPU: " + str(socket) + ", " + patternNames[pattern]) 
               
               if (usePinnedMem): 
                  #CASE 4: All sockets, all patterns, each src/dest pair, pinned source
                  plt.figure(label + "_pinned_src")
                  plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[pattern], marker = marker[socket], label = "CPU: " + str(socket) + ", " + patternNames[pattern]) 

                  #CASE 4: All sockets, all patterns, each src/dest pair, pinned destination
                  plt.figure(label + "_pinned_dest")
                  plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[pattern], marker = marker[socket], label = "CPU: " + str(socket) + ", " + patternNames[pattern]) 

                  #CASE 4: All sockets, all patterns, each src/dest pair, both pinned 
                  plt.figure(label + "_both_pinned")
                  plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[pattern], marker = marker[socket], label = "CPU: " + str(socket) + ", " + patternNames[pattern]) 

         #CASE 4: All sockets, all patterns, each src/dest pair, both pageable
         label = "all_cpu" + "_src" + str(srcNode) + "_dest" + str(destNode) + "_all_patterns"
         save_figure(label + "_both_page", "All CPUs, All Patterns, Src " + str(srcNode) + ", Dest " + str(destNode) + ", Both Pageable", label + "_both_page")
         plt.clf()         
        
         if (usePinnedMem): 
            #CASE 4: All sockets, all patterns, each src/dest pair, pinned source
            save_figure(label + "_pinned_src", "All CPUs, All Patterns, Src " + str(srcNode) + ", Dest " + str(destNode) + ", Pinned Src", label + "_pinned_src")
            plt.clf()         

            #CASE 4: All sockets, all patterns, each src/dest pair, pinned destination
            save_figure(label + "_pinned_dest", "All CPUs, All Patterns, Src " + str(srcNode) + ", Dest " + str(destNode) + ", Pinned Dest", label + "_pinned_dest")
            plt.clf()         

            #CASE 4: All sockets, all patterns, each src/dest pair, both pinned 
            save_figure(label + "_both_pinned", "All CPUs, All Patterns, Src " + str(srcNode) + ", Dest " + str(destNode) + ", Both Pinned", label + "_both_pinned")
            plt.clf()         

   #CASE 5: All sockets, each mem combination, all src/dest pairs, one pattern
   label = "all_cpu_src_dest"
   for socket in range(startSocket, startSocket + numSockets):
      for srcNode in range(0, numNodes):
         for destNode in range(0, numNodes):
            y_idx =  socket * numSockets * numNodes * numMemTypes * numPatterns + srcNode * numNodes * numMemTypes * numPatterns + destNode * numMemTypes * numPatterns
            
            #CASE 5: All sockets, both pageable, all src/dest pairs, repeated pattern
            plt.figure(label + "_both_page")
            plt.scatter(blkSize, data[y_idx + 0 * numPatterns], c = colors[srcNode * numNodes + destNode], marker=marker[socket], label = "CPU " + str(socket) + " Src " + str(srcNode) + " Dest " + str(destNode)) 
            
            if (usePinnedMem): 
               #CASE 5: All sockets, pinned src, all src/dest pairs, repeated pattern
               plt.figure(label + "_pinned_src")
               plt.scatter(blkSize, data[y_idx + 1 * numPatterns], c = colors[srcNode * numNodes + destNode], marker=marker[socket], label = "CPU " + str(socket) + " Src " + str(srcNode) + " Dest " + str(destNode)) 

               #CASE 5: All sockets, pinned dest, all src/dest pairs, repeated pattern
               plt.figure(label + "_pinned_dest")
               plt.scatter(blkSize, data[y_idx + 2 * numPatterns], c = colors[srcNode * numNodes + destNode], marker=marker[socket], label = "CPU " + str(socket) + " Src " + str(srcNode) + " Dest " + str(destNode)) 

               #CASE 5: All sockets, both pinned, all src/dest pairs, repeated pattern
               plt.figure(label + "_both_pinned")
               plt.scatter(blkSize, data[y_idx + 3 * numPatterns], c = colors[srcNode * numNodes + destNode], marker=marker[socket], label = "CPU " + str(socket) + " Src " + str(srcNode) + " Dest " + str(destNode)) 

   #CASE 5: All sockets, both pageable, all src/dest pairs, repeated pattern
   save_figure(label + "_both_page", "All CPUs, All Src/Dest Pairs, Both Pageable ", label + "_both_page")
   plt.clf()         
  
   if (usePinnedMem): 
      #CASE 5: All sockets, pinned src, all src/dest pairs, repeated pattern
      save_figure(label + "_pinned_src", "All CPUs, All Src/Dest Pairs, Pinned Src ", label + "_pinned_src")
      plt.clf()         

      #CASE 5: All sockets, pinned dest, all src/dest pairs, repeated pattern
      save_figure(label + "_pinned_dest", "All CPUs, Src/Dest Pairs, Pinned Dest", label + "_pinned_dest")
      plt.clf()         

      #CASE 5: All sockets, both pinned, all src/dest pairs, repeated pattern
      save_figure(label + "_both_pinned", "All CPUs, All Src/Dest Pairs, Both Pinned", label + "_both_pinned")
      plt.clf()         

             












