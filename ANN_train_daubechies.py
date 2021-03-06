"""
The program uses the back propagation network which is trained using about 160 images of car, motorcycle
and airplane each.The feature vector of images were extracted using the Daubechies D4 Wavelet Transform. These
vectors were used for training and testing purposes.

Referred: http://www.swarthmore.edu/NatSci/mzucker1/e27/simple_nnet_example.py

@author Aditya Pulekar, Chirag Kular
"""

import cv2,matplotlib.pyplot as plt
import numpy as np,os
import allFeatures
import pdb,math

class project:
    __slots__="params"

    def __init__(self,crit,stepSize,momentum):
        self.params=dict(term_crit=crit,train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale=\
                         stepSize,bp_moment_scale= momentum)

if __name__=='__main__':
    #For ANN
    step=0.015 ; momentum=0.0 ; total_steps=11000 ; max_allowable_err=0.0002
    condn=cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    p=project((condn,total_steps,max_allowable_err),step,momentum)

    folderCount=1;
    featureDict={}
    maxRowOfFeatureVector=float('-inf')
    maxColOfFeatureVector=float('-inf')
    while folderCount <= 3:
        featureDict[folderCount]=[]
        path="E:\Aditya\Python_RIT\FCV\Project\Database_S" + "/" + str(folderCount) + "/"  #This statement shows that even a forward slash works
        dataFolders=os.listdir(path)
        print "Length of dataFolders: ",len(dataFolders)
        for data in dataFolders:
            img_col=cv2.imread((path+str(data)))    #Its by default a color image
            img_gr=cv2.cvtColor(img_col,cv2.COLOR_BGR2GRAY)

            #Suggested to be used for resolving the "data type = 17 is not supported" error
            img_color=np.asarray(img_col[:,:])
            img_gray=np.asarray(img_gr[:,:])
            temp=allFeatures.daubechies(img_gray)
            [r,c]=[len(temp),len(temp[0])]
            if r > maxRowOfFeatureVector:
                maxRowOfFeatureVector=r
            if c > maxColOfFeatureVector:
                maxColOfFeatureVector=c
            featureDict[folderCount].append(temp.flatten())
        print "DataFolder ",str(folderCount), " done!"
        folderCount+=1
    print "All feature vectors obtained!"
    #FEATURE VECTORS OBTAINED

    index=0
    countOfImages=0
    inpt=np.empty([480,r*c],dtype=np.float32)     #Rows --> Total_Images, Column--> length of daubechies feature vector
    target=np.zeros([480,3],dtype=np.float32)  #499x3-->number of descriptors=number of images & number of groups=3


    #Since an image has many interest points and hence many descriptor vectors, should we include every such vector in "trained"
    for itr in range(3):
        print "Group ",itr+1, " done!"
        for rows in featureDict[itr+1]:
            inpt[index]=np.resize(rows,(1,r*c))
            target[index][itr]=1.0                 #So 1st column in target-> group1, 2nd column in target-> group2
                                                       #3rd column in target-> group3
            index+=1
            countOfImages+=1

    print "Total Number of Images: ",countOfImages
    print "final_trained: ",inpt.shape
    print "final_groups: ",target.shape

    total_inputs=len(inpt[0]) ; total_hidden=5 ; total_outputs=3   #len(inpt)
    ann_layers=np.array([total_inputs,total_hidden,total_outputs])
    NN=cv2.ANN_MLP(ann_layers)

    #Now we train our neural network
    iter=NN.train(np.float32(inpt),np.float32(target),None,params=p.params)   #NOTE: "params" also matter in determining the accuracy

    #NOTE: In predictions, we are categorizing every feature descriptor/ feature of the test image into one of the 3 groups
    print "Number of iterations: ",iter

    #Testing the trained system
    testPath= "E:\Aditya\Python_RIT\FCV\Project\Database_S\TestImages\Test_2"+"/"
    test_img=os.listdir(testPath)
    count=0
    print "Total test images: ", len(test_img)
    for it in test_img:
        img_TEST=cv2.imread(testPath+str(it))
        img_gray_TEST=cv2.cvtColor(img_TEST,cv2.COLOR_BGR2GRAY)
        predictn=np.ones([1,3],dtype=np.float32)
        temp=allFeatures.daubechies(img_gray_TEST)
        test_input=np.resize(temp.flatten(),(1,r*c))
        test_input=np.array(test_input,dtype=np.float32)
        NN.predict(test_input,predictn)
        #Predicting the group using the group mean
        # Here the problem is caused if all three columns of "predictn" are float('nan'). In such case
        # we get a random answer
        obj=np.argmax(predictn)    #In case of something like [2.3,4.5,nan]--> Group 3 is chosen
        if obj+1 == 1:
            im="Car"
        elif obj+1 == 2:
            im="Motorcycle"
        else:
            im="Airplane"
        print "\nImage ",count+1," contains ",im
        cv2.putText(img_TEST,im,(30,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,255),thickness=5)
        cv2.imshow("Annotated Image",img_TEST)
        cv2.waitKey(0)
        print "\n"
        count+=1
    cv2.destroyAllWindows()
    pdb.set_trace()