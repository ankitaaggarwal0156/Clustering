'''
Created on Sep 23, 2016

@author: dell
'''
import random
import math
import sys
import numpy as np
from numpy.linalg import inv

x_coordinate_list= []
y_coordinate_list= []
centroid=[]
cluster1=[]
cluster2=[]
cluster3=[]
p1=0.0
p2=0.0
p3=0.0
W,G, Mu=[],[],[]
def ReadFile():
    list_of_coordinates = []
    global x_coordinate_list
    global y_coordinate_list
    with open("./clusters.txt", "r") as fo:
        for line in fo:
            list_of_coordinates.append(line)
    fo.close()

    for line in list_of_coordinates:
        list_of_items_in_line = line.split(",")
        x_coordinate_list.append(float(list_of_items_in_line[0]))
        y_coordinate_list.append(float(list_of_items_in_line[1]))
    #return x_coordinate_list, y_coordinate_list

def CreateClusters():
    for i in range(len(x_coordinate_list)):
        distance1= math.sqrt((abs(x_coordinate_list[i] - centroid[0][0]))**2 + (abs(y_coordinate_list[i] - centroid[1][0]))**2)
        distance2= math.sqrt((abs(x_coordinate_list[i] - centroid[0][1]))**2 + (abs(y_coordinate_list[i] - centroid[1][1]))**2)
        distance3= math.sqrt((abs(x_coordinate_list[i] - centroid[0][2]))**2 + (abs(y_coordinate_list[i] - centroid[1][2]))**2)
        #min_distance= min(distance1, distance2, distance3)
        if(distance1 !=distance2 and distance2!=distance3 and distance1!=distance3):
            min_distance= min(distance1, distance2, distance3)
        elif(distance1 == distance2 and distance2 == distance3):
            min_distance = distance1
        elif(distance1==distance2 and distance3>distance1):
            min_distance = distance1
        elif(distance2==distance3 and distance1>distance2):
            min_distance = distance2
        elif(distance1==distance3 and distance2>distance1):
            min_distance = distance1
        elif(distance1==distance2 and distance3<distance1):
            min_distance = distance3
        elif(distance2==distance3 and distance1<distance2):
            min_distance = distance1
        elif(distance1==distance3 and distance2<distance1):
            min_distance = distance2
            
        #print "min distance =", min_distance
        if(min_distance==distance1):
            #Incase coordinate already present in Cluster1 then don't do anything!
            if(cluster1.__contains__([x_coordinate_list[i],y_coordinate_list[i]])):
                pass
            elif(cluster2.__contains__([x_coordinate_list[i],y_coordinate_list[i]])):
                    cluster2.remove([x_coordinate_list[i],y_coordinate_list[i]])
                # Incase coordinate was present in cluster3, first remove from cluster3
            elif (cluster3.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                    cluster3.remove([x_coordinate_list[i], y_coordinate_list[i]])
                #Finally append the coordinate in cluster1
            else:
                cluster1.append([x_coordinate_list[i],y_coordinate_list[i]])
        if (min_distance == distance2):
            if (cluster2.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                pass
            elif (cluster1.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                    cluster1.remove([x_coordinate_list[i], y_coordinate_list[i]])
            elif (cluster3.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                    cluster3.remove([x_coordinate_list[i], y_coordinate_list[i]])
            else:
                cluster2.append([x_coordinate_list[i],y_coordinate_list[i]])
        if (min_distance == distance3):
            if (cluster3.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                pass
            elif (cluster1.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                    cluster1.remove([x_coordinate_list[i], y_coordinate_list[i]])
            elif (cluster2.__contains__([x_coordinate_list[i], y_coordinate_list[i]])):
                    cluster2.remove([x_coordinate_list[i], y_coordinate_list[i]])
            else:
                cluster3.append([x_coordinate_list[i],y_coordinate_list[i]])
    return cluster1,cluster2,cluster3

def CalculateMean(cluster):
    SumX,SumY=0,0
    for i in range(len(cluster)):
        SumX += cluster[i][0]
        SumY += cluster[i][1]
    NewMean_Cluster=[float(SumX/len(cluster)),float(SumY/len(cluster))]
    #print "New Centroid: ",NewMean_Cluster
    return NewMean_Cluster

def kmeans():
    global centroid #centroid always points to current 3 centroid coordinates
    flag1,flag2,flag3=0,0,0
    oldCentroid1 = [centroid[0][0],centroid[1][0]]
    oldCentroid2 = [centroid[0][1], centroid[1][1]]
    oldCentroid3 = [centroid[0][2], centroid[1][2]]

    Clusters = CreateClusters()
    #print "cluster1", Clusters[0]
    #print "cluster2", Clusters[1]
    #print "cluster3", Clusters[2]
    centroid1 = CalculateMean(Clusters[0])
    centroid2 = CalculateMean(Clusters[1])
    centroid3 = CalculateMean(Clusters[2])

    if (centroid1==oldCentroid1):
        #print "centroid1:",centroid1
        flag1=1
    else:
        flag1=0
        oldCentroid1=centroid1
        centroid[0][0]=centroid1[0]
        centroid[1][0]=centroid1[1]

    if (centroid2==oldCentroid2):
        #print "centroid2:",centroid2
        flag2=1
    else:
        flag2=0
        oldCentroid2=centroid2
        centroid[0][1]=centroid2[0]
        centroid[1][1]=centroid2[1]

    if (centroid3==oldCentroid3):
        #print "centroid3:",centroid3
        flag3=1
    else:
        flag3=0
        oldCentroid3=centroid3
        centroid[0][2]=centroid3[0]
        centroid[1][2]=centroid3[1]
    if (flag1==1 and flag2==1 and flag3==1):
        print "---------------KMeans Output-------------"
        print "*****************"
        #print "Cluster1: ", cluster1
        print "Length of Cluster 1:", len(cluster1)
        print "Centroid Cluster 1:", centroid1
        print"******************"
        #print "Cluster2", cluster2
        print "Length of Cluster2: ", len(cluster2)
        print "Centroid Cluster2: ", centroid2
        print"******************"
        #print "Cluster3:", cluster3
        print "Length of Cluster 3:", len(cluster3)
        print "Centroid Cluster 3:", centroid3
        print "-----------------------------------------"
    else:
        kmeans()
        
        
def Gj(A, B, C):
    x = ((-1 / 2) * A * np.linalg.inv(C) * B)
    #print "x:", x
    G = (float)(1 / math.sqrt((2 * math.pi) ** 2 * np.linalg.det(C))) * math.exp(x)
    #print G
    return G 

       
def amplitude(j):
    global p1,p2,p3,G
    W=[]
    for i in range(len(x_coordinate_list)):
        W.append((p1 * G[j][i]) / (p1 * G[0][i]+ p2* G[1][i]+ p3* G[2][i]))
    #print W
    return W


def sumAmplitude(j):
    sumW=0.0
    for i in W[j]:
        sumW+=i
    #print sumW
    return sumW
        
def GMM_cluster(cluster,Mean, CoV):
    G_cluster = []
    A = []
    B = []
    cluster = CalculateTranspose(cluster)   #Our input list is of dimension 150 X2, transforming it into 2X150
    for i in range(len(x_coordinate_list)):
        A = [x_coordinate_list[i] - Mean[0], y_coordinate_list[i] - Mean[1]]
        A = np.matrix(A)
        B = A.getT()
        G_cluster.append(Gj(A, B, CoV))
    return G_cluster


def CalculateD(cluster,Mean):
    D = []
    for i in range(len(x_coordinate_list)):
        D.append([x_coordinate_list[i] - Mean[0], y_coordinate_list[i] - Mean[1]])
    return D

def CalculateTranspose(matrix):
    matrix_Transpose = [list(i) for i in zip(*matrix)]
    #print "DTranspose", matrix_Transpose
    return matrix_Transpose     

def MeanMaximization(j):
    Mu1=[]
    mX=0.0
    mY=0.0
    global x_coordinate_list, y_coordinate_list,W
    for i in range(len(W[j])):
        mX=mX+ (W[j][i] * x_coordinate_list[i])
        mY = mY + (W[j][i] * y_coordinate_list[i])       
    Mu1=[mX/sumAmplitude(j),mY/sumAmplitude(j)]
    return Mu1 


def CovMaximization(j):
    X=[[0,0],[0,0]]
    CoMax=[]
    X=np.matrix(X)
    H=[]
    HT=[]
    global x_coordinate_list, y_coordinate_list,W, Mu1,Mu2,Mu3
    for i in range(len(W[j])):
        #W[j][i]*
        H=[x_coordinate_list[i] - Mu[j][0], y_coordinate_list[i] - Mu[j][1]]
        H=np.matrix(H)
        HT= H.getT()
        HT=np.matrix(HT)
        X1=W[j][i]*HT*H
        X1=np.matrix(X1)
        X=X+X1
    CoMax=X/sumAmplitude(j)
    return CoMax
#Program Start

def main(cluster1, cluster2, cluster3,Mean1,Mean2, Mean3, CoV1, CoV2, CoV3):
    global W
    global G
    global p1,p2,p3, Mu, CovM1, CovM2,CovM3
    #print "prior prob =", p1, p2, p3
    G.append(GMM_cluster(cluster1,Mean1, CoV1))
    G.append(GMM_cluster(cluster2,Mean2, CoV2))
    G.append(GMM_cluster(cluster3,Mean3, CoV3))
    W.append(amplitude(0))
    W.append(amplitude(1))
    W.append(amplitude(2))
    
    #Maximization step
    p1M=sumAmplitude(0)/150
    p2M=sumAmplitude(1)/150
    p3M=sumAmplitude(2)/150
    Mu.append(MeanMaximization(0))
    Mu.append(MeanMaximization(1))
    Mu.append(MeanMaximization(2))
    CovM1=CovMaximization(0)
    CovM2=CovMaximization(1)
    CovM3=CovMaximization(2)
    
    ch1=checkListEqual(Mu[0], Mean1)
    ch2=checkListEqual(Mu[1], Mean2)
    ch3=checkListEqual(Mu[2], Mean3)
    if(ch1!=0 and ch2!=0 and ch3!=0):
        print " --------------------GMM Output -------------------"
        print "Final Mean of Gaussian1 ", Mean1
        print "Final Amplitude of Gaussian1 ", p1M
        print "Covariance Matrix ",CovM1
        print "********"
        print "Final Mean of Gaussian2 ", Mean2
        print "Final Amplitude of Gaussian2 ", p2M
        print "Covariance Matrix ",CovM2
        print "********"
        print "Final Mean of Gaussian3 ", Mean3
        print "Final Amplitude of Gaussian3 ", p3M
        print "Covariance Matrix ",CovM3
        print "********"
        print "---------------------------------------------------"
        
    else:
        p1=p1M
        p2=p2M
        p3=p3M
        Mean1=Mu[0]
        Mean2=Mu[1]
        Mean3=Mu[2]
        CoV1=CovM1
        CoV2=CovM2
        CoV3=CovM3
        main(cluster1, cluster2, cluster3,Mean1,Mean2,Mean3, CoV1, CoV2, CoV3)
        
    
    

def checkListEqual(Mu, Mean1):
    Q,N,dec=[],[],[]
    flag=[]
    #print"Mu", np.array(Mu)
    #print "Mean1*0.97",np.multiply(0.97,Mean1)
    Q = np.array(Mu) >np.array(np.multiply(0.97,Mean1)) 
    #print Q
    if False in Q:
        flag.append(False)
    else:
        flag.append(True)
    flag.append(np.array_equal(Mu, np.multiply(0.97,Mean1)))
    #print "Mean1*1.03",np.multiply(1.03,Mean1)
    N = np.array(Mu) <np.array(np.multiply(1.03,Mean1)) 
    #print N
    if False in N:
        flag.append(False)
    else:
        flag.append(True)
    flag.append(np.array_equal(Mu, np.multiply(1.03,Mean1)))
    flag.append(np.array_equal(Mu, Mean1))
    #print flag
    if(flag[0]==True or flag[1]==True ):
        dec.append(True)
    else:
        dec.append(False)
    if(flag[2]==True or flag[3]==True ):
        dec.append(True)
    else:
        dec.append(False)   
    dec.append(flag[4])
    #print dec   
    
    if ((dec[0]==True and dec[1]==True)or(dec[2]==True)):
        return 1 #True
    else:
        return 0

ReadFile()
centroid = [random.sample(x_coordinate_list, 3), random.sample(y_coordinate_list, 3)]
kmeans()
p1 = (float)(len(cluster1)) / (float)(len(x_coordinate_list))
p2 = (float)(len(cluster2)) / (float)(len(x_coordinate_list))
p3 = (float)(len(cluster3)) / (float)(len(x_coordinate_list))
Mean1 = CalculateMean(cluster1)
Mean2 = CalculateMean(cluster3)
Mean3 = CalculateMean(cluster2)
CoVT1 = CalculateD(cluster1,Mean1)
CoV1 = CalculateTranspose(CoVT1)
D1=np.cov(CoV1)
CoVT2 = CalculateD(cluster2,Mean2)
CoV2 = CalculateTranspose(CoVT2)
D2=np.cov(CoV2)
CoVT3 = CalculateD(cluster3,Mean3)
CoV3 = CalculateTranspose(CoVT3)
D3=np.cov(CoV3)
main(cluster1, cluster2, cluster3,Mean1,Mean2,Mean3,D1, D2, D3)