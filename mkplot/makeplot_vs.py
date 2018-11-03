from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import numpy as np
from sklearn import metrics

def plot(vsig, vbkg, num_bins, title):
  fig, ax = plt.subplots()
  sn1, sbins, patches = ax.hist(vsig, num_bins, range=[0.0, 1.0], alpha=0.5, normed=True, color="red", label="signal")
  bn1, bbins, patches = ax.hist(vbkg, num_bins, range=[0.0, 1.0], alpha=0.5, normed=True, color="blue", label="background")

  #print sbins, " ", bbins
  # add a 'best fit' line 

  ax.set_xlabel('DNN output')
  ax.set_ylabel('Entries')
  ax.set_title('Deep learnging output')

  #log scale
  #plt.yscale("log")

  # Tweak spacing to prevent clipping of ylabel
  fig.tight_layout()
  plt.legend(loc='upper left')
  #plt.show()
  print("output is saved!")

  fig.savefig(title+".pdf")

  return sn1, bn1

def split_data( data ):
  tmp_sig = []
  tmp_bkg = []
  for row in data:
    x = float(str.split(row[0]," ")[0])
    tag = float(str.split(row[0]," ")[1])
    if tag == 1:
      tmp_sig.append( x )
    elif tag == 0:
      tmp_bkg.append( x )

  return tmp_sig, tmp_bkg

output_layer10_100 = csv.reader(open('materials-20to120/output.csv','r'))
output_rel = csv.reader(open('materials-20to120/reliso_20to120.csv'))

value_sig = [[],[]]
value_bkg = [[],[]]

value_sig[0], value_bkg[0] = split_data( output_layer10_100 )
value_sig[1], value_bkg[1] = split_data( output_rel )

num_bins = 100
 
sig_eff = [[],[]]
bkg_eff = [[],[]]

# the histogram of the data
sn0, bn0 = plot( value_sig[0], value_bkg[0], num_bins, "output_layer10_100")
sn1, bn1 = plot( value_sig[1], value_bkg[1], num_bins, "output_rel")

fig, ax = plt.subplots()

# Efficiency
for i in range(0,100):
  sig_eff[0].append(sum(sn0[i:num_bins]) / sum(sn0[0:num_bins]))
  bkg_eff[0].append(sum(bn0[i:num_bins]) / sum(bn0[0:num_bins]))

for i in range(0,100):
  sig_eff[1].append(sum(sn1[0:i]) / sum(sn1[0:num_bins]))
  bkg_eff[1].append(sum(bn1[0:i]) / sum(bn1[0:num_bins]))

for i in range(0,2):
 auc = metrics.auc(bkg_eff[i], sig_eff[i])
 print ("auc : ", auc, "\nlayer : ", i+1)

ax = plt.plot(bkg_eff[0], sig_eff[0], label='Deep Iso')
ax = plt.plot(bkg_eff[1], sig_eff[1], label='Relative Iso')

plt.ylabel("Signal efficiency (%)")
plt.xlabel("Background efficiency (%)")
plt.legend(loc='center right')
plt.savefig("roc_vs.pdf")
print("roc curve is saved!")
#plt.show()

