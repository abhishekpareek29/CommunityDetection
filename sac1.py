import pandas as pd
import numpy as np
import sys
from igraph import *
from scipy import spatial

simMatrix = []
simMatrix2 = []

# Implements phase1.
def phase1(g, alpha, C):
	V = len(g.vs)
	m = len(g.es)
	iteration = 0
	check = 0

	while (check == 0 and iteration < 15):
		check = 1
		for vi in range(V):
			maxV = -1
			maxDeltaQ = 0.0
			clusters = list(set(C))
			for vj in clusters:
				if (C[vi] != vj):
					dQ = (alpha*change_mod(C, g, vi, vj)) + ((1-alpha)*change_attr_sim(C, g, vi, vj))
					if (dQ > maxDeltaQ):
						maxDeltaQ = dQ
						maxV = vj
			if (maxDeltaQ > 0.0 and maxV != -1):
				check = 0
				C[vi] = maxV
		iteration = iteration + 1
	return C

# Implements phase2.
def phase2(g, C):
	newC = sequential_clusters(C)
	temp = list(Clustering(newC))
	L = len(set(newC))
	simMatrix = np.zeros((L,L))

	for i in range(L):
		for j in range(L):
			similarity = 0.0
			for k in temp[i]:
				for l in temp[j]:
					similarity = similarity + simMatrix2[k][l]
			simMatrix[i][j] = similarity

	g.contract_vertices(newC)
	g.simplify(combine_edges=sum)
	return

# Make the clusters sequential.
def sequential_clusters(C):
	mapping = {}
	newC = []
	c = 0
	for i in C:
		if i in mapping:
			newC.append(mapping[i])
		else:
			newC.append(c)
			mapping[i] = c
			c = c + 1
	return newC

# Change in modularity.
def change_mod(C, g, v1, v2):
	Q1 = g.modularity(C, weights='weight')
	temp = C[v1]
	C[v1] = v2
	Q2 = g.modularity(C, weights='weight')
	C[v1] = temp
	return (Q2-Q1);

# Change in attribute similarity.
def change_attr_sim(C, g, v1, v2):
	S = 0.0;
	indices = [i for i, x in enumerate(C) if x == v2]
	for v in indices:
		S = S + simMatrix[v1][v]
	return S/(len(indices)*len(set(C)))

# Total attribute similarity.
def total_attr_sim(C, g):
	clusters = list(Clustering(C))
	V = g.vcount()
	S = 0.0
	for c in clusters:
		T = 0.0
		for v1 in c:
			for v2 in C:
				if (v1 != v2):
					T = T + simMatrix[v1][v2]
		T = T/len(c)
		S = S + T
	return S/(len(set(C)))

def main(alpha):
	# Create igraph.
	attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')

	V = len(attributes)

	with open('data/fb_caltech_small_edgelist.txt') as f:
		edges = f.readlines()
	edges = [tuple([int(x) for x in line.strip().split(" ")]) for line in edges]

	g = Graph()
	g.add_vertices(V)
	g.add_edges(edges)
	g.es['weight'] = [1]*len(edges)

	for col in attributes.keys():
		g.vs[col] = attributes[col]

	# Pre-Computing Similarity Matrix
	global simMatrix
	global simMatrix2
	simMatrix = np.zeros((V,V))
	for i in range(V):
		for j in range(V):
			simMatrix[i][j] = 1 - spatial.distance.cosine(list(g.vs[i].attributes().values()), list(g.vs[j].attributes().values()))

	# Copy.
	simMatrix2 = np.array(simMatrix)

	# Run Algorithm
	V = g.vcount()
	C = phase1(g, alpha, list(range(V)))
	print('Number of Communities after Phase 1')
	print(len(set(C)))
	C = sequential_clusters(C)

	# Modularity of phase 1 clustering
	mod1 = g.modularity(C, weights='weight') + total_attr_sim(C, g)

	# Phase 2
	phase2(g, C)

	# Run phase 1 again.
	V = g.vcount()
	C2 = phase1(g, alpha, list(range(V)))
	C2new = sequential_clusters(C2)
	clustersPhase2 = list(Clustering(C2new))
	#Composite modularity of contracted clustering
	mod2 = g.modularity(C, weights='weight') + total_attr_sim(C, g)

	finalC = []
	C1new = sequential_clusters(C)
	clustersPhase1 = list(Clustering(C1new))

	# Mapping the super clusters from phase 2 to original vertices.
	for c in clustersPhase2:
		t = []
		for v in c:
			t.extend(clustersPhase1[v])
		finalC.append(t)


	# Write clusters to output file.
	if alpha == 0.5:
		alpha = str(5)
	elif alpha == 0.0:
		alpha = str(0)
	elif alpha == 1.0:
		alpha = str(1)

	if (mod1 > mod2):
		clusters = clustersPhase1
	else:
		clusters = clustersPhase2

	file = open("communities_" + alpha + ".txt", 'w+')
	for c in clusters:
		for i in range(len(c)-1):
			file.write("%s," % c[i])
		file.write(str(c[-1]))
		file.write('\n')
	file.close()

	return clusters


if __name__ == "__main__":
	if len(sys.argv) <= 1:
		print("Missing argument.")
	else:
		main(float(sys.argv[1]))
