# Clustering
K-means and GMM

To implement the K-Means algorithm and Expectation Maximization Algorithm for clustering using a Gaussian Mixture Model. 

PROJECT DESCRIPTION
Programming language used :       Python
Data Structure            :       Lists, Matrix
File Name                 :       Cluster_K_GMM.py
Inputs                    :       clusters.txt, set of 2dimensional - 150 data points
Output                    :       KMeans – centroid of each cluster, and its length
                                  GMM – Mean, Amplitude, Co-Variance of the probability distribution

Implementation:

kmeans(): recursive function, that is used as the caller for :
-	creating clusters, based on the mean value, randomly generated
-	calculating mean for the new clusters
-	checking if the old mean is the same as new mean of cluster
-	if the means are same, therefore, no further cluster modifications are required and we have reached the optimum solution
-	if means are not same, kmeans performs the computation again, with the new mean values.



