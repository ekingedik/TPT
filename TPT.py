# -*- coding: utf-8 -*-

import os
import cPickle as pickle
import numpy as np
from scipy import signal
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from helperFuncs import ComputeKernelEMD
from helperFuncs import ComputeKernelEMD1D
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from emd import emd

## Compute the EMD kernel
## X1 and X2 are lists with data from multiple participants as elements
def ComputeKernelEMD(X1,X2,dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1,sz2))
    for i in range(0,sz1):
        for j in range(i,sz2):
            print i,j
            startT = time.time()
            D[i,j] = (emd(X1[i],X2[j],distance=dist))
            endT = time.time() - startT
            print 'EMD took ' + str(endT) + ' seconds.'
    D = D + np.transpose(np.triu(D,k=1))
    #D2 = np.exp((-1/np.mean(D[np.nonzero(D)]))*D)
    return D

## Compute the EMD kernel when X1 only has one element
def ComputeKernelEMD1D(X1,X2,dist):
    sz1 = np.shape(X1)[0]
    sz2 = np.shape(X2)[0]
    D = np.zeros((sz1,sz2))
    for i in range(0,sz1):
        for j in range(i,sz2):
            print i,j
            startT = time.time()
            D[i,j] = (emd(X1[i],X2[j],distance=dist))
            endT = time.time() - startT
            #print 'EMD took ' + str(endT) + ' seconds.'
    #D2 = np.exp((-1/np.mean(D[np.nonzero(D)]))*D)
    return D

# Function to extract statistical and spectral features
def FeatureExtractor2(data,sz):
    sz = (sz/2)+1
    if data.ndim == 1:
        # Compute psd with periodogram
        f,psd = signal.periodogram(data,fs=20,nfft=800)
        ids = np.nonzero(np.logical_or.reduce((f==0,f==0.125, f == 0.25,f == 0.5, f== 1, f== 2, f==4, f==8))) 
        # Get 8 bins, log-spaced
        psd = psd[ids[0]]
        # Compute mean and variance
        meanCur = np.mean(data,axis=0)
        varCur = np.std(data,axis=0)
    else:
        psd =np.empty((8,3))
        meanCur = np.empty((1,3))
        varCur =np.empty((1,3))
        # For each axis,
        for i in range(0,3):
            dataCur = data[:,i]
            # Compute psd with periodogram
            f,psd1 = signal.periodogram(dataCur,fs=20,nfft=800)
            ids = np.nonzero(np.logical_or.reduce((f==0,f==0.125, f == 0.25,f == 0.5, f== 1, f== 2, f==4, f==8)))
            # Get 8 bins, log-spaced
            psd[:,i] = psd1[ids[0]]
            # Compute mean and variance
            meanCur[:,i] = np.mean(dataCur,axis=0)
            varCur[:,i] = np.std(dataCur,axis=0) 
        psd = psd.reshape(8*3)
        meanCur = meanCur.reshape(3)
        varCur = varCur.reshape(3)
    return psd,meanCur,varCur   


## Walk through the train folder, read all filenames and sort them
for root, dirnames, filenames in os.walk('./train'):
    print filenames
filenames.sort()

# Lists for preserving acceleration magnitude, axes of acceleration, absolute accel and annotations
# Lists are size of N, number of participants
allMag = []
allAccel = []
allAccelAbs = []    
allAnnotations = []

# Read the annotations from the csv file. This file includes speaking annotations for all participants
# Each column corresponds to a participants (or vice versa, check the shape)
labels =np.genfromtxt('Train_labels.csv',delimiter=',')
# For each participant
for curid,cur in enumerate(filenames):
    # Read the acceleration csv file
    curAccel = np.genfromtxt('./train/'+cur,delimiter=',')
    # First column is timestamps, dont' use.
    # Normalize each axis
    accelX = stats.zscore(curAccel[:,1])
    accelY = stats.zscore(curAccel[:,2])
    accelZ = stats.zscore(curAccel[:,3])
    # Get absolute values
    absAccelX = np.absolute(accelX)
    absAccelY = np.absolute(accelY)
    absAccelZ = np.absolute(accelZ)
    # Compute magnitude
    accelMag = np.sqrt(pow(accelX,2) + pow(accelY,2) + pow(accelZ,2))
    # Save normalised and absolute values in 3D arrays
    accelFull = np.transpose(np.array([accelX,accelY,accelZ]))
    accelAbsFull= np.transpose(np.array([absAccelX,absAccelY,absAccelZ]))       
    # Append to lists
    allMag.append(accelMag)
    allAccel.append(accelFull)
    allAccelAbs.append(accelAbsFull)   
    allAnnotations.append(labels[:,curid])


# Window size in secs.
windowSize = 3
sampleSize = windowSize*20


# Use feature extractor function to form feature vectors for each participant
# Save them in lists: features variable for features and y_spk for the speaking annotations
y_spk = []
features = []

for curParticipant,(magChalc,accelAll,accelAbsAll,spkAnn) in enumerate(zip(allMag,allAccel,allAccelAbs,allAnnotations)):
    print curParticipant
    spkAnnCur = []
    featuresCur = []
    sz = np.shape(magChalc)[0]
    # Use the sliding window of 3 seconds with 1.5s overlap
    for i in range(0,sz-sampleSize,sampleSize/2):
        # Get the current magnitude, accel and absaccel
        curMag = magChalc[i:i+sampleSize]
        curAccel = accelAll[i:i+sampleSize,:]
        curAbsAccel = accelAbsAll[i:i+sampleSize,:]
        # Extract the features from each
        psd1_Mag,meanCur_Mag,varCur_Mag = FeatureExtractor2(curMag,np.shape(curAccel)[0])    
        psd1_XYZ,meanCur_XYZ,varCur_XYZ = FeatureExtractor2(curAccel,np.shape(curAccel)[0])    
        psd1_XYZAbs,meanCur_XYZAbs,varCur_XYZAbs= FeatureExtractor2(curAbsAccel,np.shape(curAccel)[0])
        # Form the feature vector for this slice and append it to the list
        featuresAll = np.hstack((psd1_Mag,meanCur_Mag,varCur_Mag,psd1_XYZ,meanCur_XYZ,varCur_XYZ,psd1_XYZAbs,meanCur_XYZAbs,varCur_XYZAbs))
        featuresCur.append(featuresAll)
        # Get the annotation for this slice and use majority voting to decide if positive or negative
        curSpk = np.asarray(spkAnn[i:i+sampleSize])
        if curSpk.sum() > sampleSize/2:
            spkAnnCur.append(1)
        else:
            spkAnnCur.append(0)
    # Append features for all slices for this participant
    y_spk.append(np.asarray(spkAnnCur))
    features.append(np.asarray(featuresCur))


# Get features as array
X = np.vstack((features))   
y = np.hstack((y_spk))

########################################################################################################################
########################################################################################################################

## Do the same procedure for the test set
for root, dirnames, filenames in os.walk('./test'):
    print filenames
filenames.sort()

allMagTest = []
allAccelTest = []
allAccelAbsTest = []    
allAnnotationsTest = []

labelsTest =np.genfromtxt('Test_labels.csv',delimiter=',')  

for curid,cur in enumerate(filenames):   
    curAccel = np.genfromtxt('./test/'+cur,delimiter=',')
    accelX = stats.zscore(curAccel[:,1])
    accelY = stats.zscore(curAccel[:,2])
    accelZ = stats.zscore(curAccel[:,3])
    absAccelX = np.absolute(accelX)
    absAccelY = np.absolute(accelY)
    absAccelZ = np.absolute(accelZ)    
    accelMag = np.sqrt(pow(accelX,2) + pow(accelY,2) + pow(accelZ,2))     
    accelFull = np.transpose(np.array([accelX,accelY,accelZ]))    
    accelAbsFull= np.transpose(np.array([absAccelX,absAccelY,absAccelZ]))       
    allMagTest.append(accelMag)
    allAccelTest.append(accelFull)
    allAccelAbsTest.append(accelAbsFull)   
    allAnnotationsTest.append(labelsTest[:,curid])

y_spk_test = []
features_test = []
for curParticipant,(magChalc,accelAll,accelAbsAll,spkAnn) in enumerate(zip(allMagTest,
                   allAccelTest,allAccelAbsTest,allAnnotationsTest)):
    print curParticipant
    spkAnnCur = []
    featuresCur = []
    sz = np.shape(magChalc)[0]
    for i in range(0,sz-sampleSize,sampleSize/2):
        curMag = magChalc[i:i+sampleSize]
        curAccel = accelAll[i:i+sampleSize,:]
        curAbsAccel = accelAbsAll[i:i+sampleSize,:]      
        psd1_Mag,meanCur_Mag,varCur_Mag = FeatureExtractor2(curMag,np.shape(curAccel)[0])    
        psd1_XYZ,meanCur_XYZ,varCur_XYZ = FeatureExtractor2(curAccel,np.shape(curAccel)[0])    
        psd1_XYZAbs,meanCur_XYZAbs,varCur_XYZAbs= FeatureExtractor2(curAbsAccel,np.shape(curAccel)[0])    
        featuresAll = np.hstack((psd1_Mag,meanCur_Mag,varCur_Mag,psd1_XYZ,meanCur_XYZ,varCur_XYZ,psd1_XYZAbs,meanCur_XYZAbs,varCur_XYZAbs))
        featuresCur.append(featuresAll)

        curSpk = np.asarray(spkAnn[i:i+sampleSize])
        if curSpk.sum() > sampleSize/2:
            spkAnnCur.append(1)
        else:
            spkAnnCur.append(0)                
    y_spk_test.append(np.asarray(spkAnnCur))
    features_test.append(np.asarray(featuresCur))

########################################################################################################################
## Train and test on the training set (As baseline)
########################################################################################################################
ss = StandardScaler()

## Normalise all the data from all participants
X = ss.fit_transform(X)

## Find the best regularisation parameter with respect to AUC  and fit on the data
lr = LogisticRegressionCV(cv=3,scoring='roc_auc', class_weight='balanced')
lr.fit(X,y)
y_out_tr = lr.predict_proba(X)

# Print the performance on the training set
print roc_auc_score(y,y_out_tr[:,1])

########################################################################################################################
## Test on the test set (As baseline)
########################################################################################################################
X_tst_all = np.vstack((features_test))   
y_tst_all = np.hstack((y_spk_test))
X_tst_all= ss.transform(X_tst_all)
y_out_tst_all = lr.predict_proba(X_tst_all)
print roc_auc_score(y_tst_all,y_out_tst_all[:,1])

########################################################################################################################
## Get the performance on each participant in the test set (As baseline)
########################################################################################################################
allAucs = []
for curX,cury in zip(features_test,y_spk_test):
    curX = np.asarray(curX)
    cury = np.asarray(cury)
    curX = ss.transform(curX)
    y_out_cur= lr.predict_proba(curX)
    allAucs.append(roc_auc_score(cury,y_out_cur[:,1]))

########################################################################################################################
## Transductive Parameter Transfer (TPT)
########################################################################################################################

# Number of subjects in the training set
numSubject = 54
featureVecs = []
filtered_XAll = []
filtered_yAll = []

#Normalise the data of each participant in the training set, separately.
#Save it in the variable featureVecs
for i in range(0,numSubject):   
    curFeat = features[i]   
    featuresAnn2 = np.copy(y_spk[i])
    ss=StandardScaler()       
    curFeat = ss.fit_transform(curFeat)               
    filtered_XAll.append(curFeat)
    filtered_yAll.append(featuresAnn2)
    featureVecs.append(curFeat)

# Using the featureVecs, compute the kernel matrix using EMD

kernel = ComputeKernelEMD(featureVecs,featureVecs,'sqeuclidean')

# Save the kernel matrix so it can be used later. (After 1 run, just read the saved kernel matrix)
with open('EMDKernel-Train.pickle', 'wb') as f:
    pickle.dump([kernel], f)


# For each participant in the training set, train a classifier on its data and get its parameters
# W (regression coefficients) and c (intercept)
# Save it in the variable params
# Params will have 54 samples, where each sample is 72D (71 for features, 1 for intercept)

numFeatures = 71
numSubCur = np.shape(kernel)[0]
params = np.zeros((numSubCur,numFeatures))   
auc_Spec = np.zeros(numSubCur)
for i in range(0,numSubCur):
    X = filtered_XAll[i]
    y = filtered_yAll[i]
    svc = LogisticRegressionCV(solver='sag',class_weight = 'balanced',scoring='roc_auc').fit(X, y)
    yhat = svc.predict_proba(X)[:,1]
    auc_Spec[i] = roc_auc_score(y,yhat)  
    print "Person dependent AUC score for participant " + str(i) + ' : '+ str(auc_Spec[i])
    coefs = svc.coef_
    params[i,:] = np.append(coefs,svc.intercept_)


# For each participant in the test set, learn the optimal parameters using the kernel matrix and params
performances= []
outs = []
for X_tst,y_tst in zip(features_test,y_spk_test) :

    # Normalise the data of the current participant
    X_tst = ss.fit_transform(X_tst)

    # Compute the distance of the current test participants dist to all training participants dists
    kernel_test = ComputeKernelEMD1D([X_tst],featureVecs,'sqeuclidean') 

    # This part is for normalising the kernel
    # We basically add the newly computed test distributions to the training sets kernel matrix
    all_kernel = np.zeros((numSubCur+1,numSubCur+1))
    all_kernel[0,0] = 0
    all_kernel[0,1:] = kernel_test
    all_kernel[1:,0] = (kernel_test)
    all_kernel[1:,1:] = kernel
    all_kernel2 =  np.exp((-1/np.mean(all_kernel[np.nonzero(all_kernel)]))*all_kernel)   

    ## Fit a ridge regressor to the training kernel matrix and params
    alphas = (2*np.logspace(-15, 15, 30,base=2))**-1
    krr = KernelRidge(kernel='precomputed')
    clf = GridSearchCV(estimator=krr, param_grid=dict(alpha=alphas),cv=5,scoring='neg_mean_absolute_error')        
    clf.fit(all_kernel[1:,1:], params)
    alp = clf.best_params_['alpha']       
    krr = KernelRidge(kernel='precomputed',alpha=alp).fit(all_kernel[1:,1:], params)

    ## Predict the parameters for the current test participant using the trained regressor
    coefTestKRR = krr.predict(all_kernel[0,1:].reshape(1, -1)) 

    ## Compute w.x_tst+intercept
    yhatKRR = coefTestKRR[:,0:numFeatures-1].dot(np.transpose(X_tst)) + coefTestKRR[:,-1]

    ## Compute the performance
    performances.append(roc_auc_score(y_tst,np.reshape(yhatKRR,np.size(y_tst))))
    print roc_auc_score(y_tst,np.reshape(yhatKRR,np.size(y_tst)))
    outs.append(yhatKRR)

## Save the results
with open('TPTResults.pickle', 'wb') as f:
    pickle.dump([performances,outs], f)



