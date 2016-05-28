"""
The program uses a Support Vector Machine (SVM) which is trained using about 160 images of car, motorcycle
and airplane each. The interest points and their subsequent feature vectors for images were extracted using the algorithms
Harris Corner Detection and Histogram of Oriented Gradients (HoG) respectively. These
feature vectors were used for training and testing purposes.

Referred: http://www.programcreek.com/python/example/70447/cv2.SVM

@author Aditya Pulekar, Chirag Kular
"""
import cv2,matplotlib.pyplot as plt
import numpy as np,os
import allFeatures
import pdb

class project:
    __slots__="params","model"

    def __init__(self):
        self.params=dict(kernel_type=cv2.SVM_RBF,svm_type=cv2.SVM_C_SVC,C=1,gamma=0.5)
        self.model=cv2.SVM()

if __name__=='__main__':
    folderCount=1;
    featureDict={}
    maxRowOfFeatureVector=float('-inf')
    maxColOfFeatureVector=float('-inf')
    while folderCount <= 3:
        featureDict[folderCount]=[]
        path="E:\Aditya\Python_RIT\FCV\Project\Database_S" + "/" + str(folderCount) + "/"
        dataFolders=os.listdir(path)
        print "Length of dataFolders: ",len(dataFolders)
        for data in dataFolders:
            img_col=cv2.imread((path+str(data)))    #Its by default a color image
            img_gr=cv2.cvtColor(img_col,cv2.COLOR_BGR2GRAY)

            #Suggested to be used for resolving the "data type = 17 is not supported" error
            img_color=np.asarray(img_col[:,:])
            img_gray=np.asarray(img_gr[:,:])
            temp=allFeatures.generateHOG(img_color,img_gray)
            [r,c]=[len(temp),len(temp[0])]
            if r > maxRowOfFeatureVector:
                maxRowOfFeatureVector=r
            if c > maxColOfFeatureVector:
                maxColOfFeatureVector=c
            featureDict[folderCount].append(temp)
        print "DataFolder ",str(folderCount), " done!"
        folderCount+=1
    print "All feature vectors obtained!"
    #FEATURE VECTORS OBTAINED

    index=0
    countOfImages=0

    final_trained=np.empty([48000,128],dtype=np.float32)
    final_groups=np.empty([48000,1],dtype=np.int)
    #Since an image has many interest points and hence many descriptor vectors, should we include every such vector in "trained"
    for itr in range(3):
        print "Group ",itr+1, " done!"
        for itr2 in range(len(featureDict[itr+1])):
            for rows in featureDict[itr+1][itr2]:
                final_trained[index]=np.resize(rows,(1,128))
                final_groups[index]=itr+1
                index+=1
            countOfImages+=1

    #NOW, We train the SVM
    p=project()
    svm=cv2.SVM()
    # SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    svm.train(np.float32(final_trained),final_groups,params=p.params)    #Try removing np.array()
    svm.save("trained_Data_160SamplesEach.dat")     #This should save the ".dat" file in the current folder

#Technique (This is exactly how I have done):...This is also how it was recommended to be done
#float data with integers labels
#1 row--> 1 descriptor and 1 group per 1 row