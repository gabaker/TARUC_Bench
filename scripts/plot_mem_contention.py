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

print "\nPlotting results from file " + text.italic + text.bold + text.red + sys.argv[1] + text.end + " given parameters:"
# + text.italic + text.bold + text.red + sys.argv[1] + text.end + 

results = open(sys.argv[1])
testParams = []
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));
testParams.append(results.readline().strip().split(","));

numNodes = int(testParams[0][0])
numGPUs = int(testParams[0][1])
blkSize = int(testParams[0][2])






