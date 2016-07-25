import scipy
import math
import numpy as np
import pandas as pd
import plotly.plotly as py
import os.path
import sys

from time import time
from sklearn import preprocessing, metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import KFold

fname = str(raw_input('Please enter the input file name containing total dataset and descriptors\n '))
if os.path.isfile(fname) :
	SubFeAll = pd.read_csv(fname, sep=",")
else:
	sys.exit("ERROR: input file does not exist")
	
#SubFeAll = pd.read_csv(fname, sep=",")
SubFeAll = SubFeAll.fillna(SubFeAll.mean()) # replace the NA values with the mean of the descriptor
header = SubFeAll.columns.values # Ues the column headers as the descriptor labels
SubFeAll.head()

# Set the numpy global random number seed (similar effect to random_state) 
np.random.seed(1)  

# Random Forest results initialised
RFr2 = []
RFmse = []
RFrmse = []

# Support Vector Regression results initialised
SVRr2 = []
SVRmse = []
SVRrmse = []

# Partial Least Squares results initialised
PLSr2 = []
PLSmse = []
PLSrmse = []

# Predictions results initialised 
RFpredictions = []
SVRpredictions = []
PLSpredictions = []

metcount = 0

# Give the array from pandas to numpy
npArray = np.array(SubFeAll)
print header.shape
npheader = np.array(header[1:-1])
print("Array shape X = %d, Y = %d " % (npArray.shape))
datax, datay =  npArray.shape

# Print specific nparray values to check the data
print("The first element of the input data set, as a minial check please ensure this is as expected = %s" % npArray[0,0])

# Split the data into: names labels of the molecules ; y the True results ; X the descriptors for each data point
names = npArray[:,0]
X = npArray[:,1:-1].astype(float)
y = npArray[:,-1] .astype(float)
X = preprocessing.scale(X)
print X.shape

# Open output files
train_name = "Training.csv"
test_name = "Predictions.csv"
fi_name = "Feature_importance.csv"

with open(train_name,'w') as ftrain:
        ftrain.write("Code originally created by James L. McDonagh 2016 for use in predicting sublimation thermodynamics,\n")
        ftrain.write("This file contains the training information for all three models (Random Forest, Support Vector Regression and Partial Least Squares),\n")
        ftrain.write("The code use a ten fold cross validation 90% training 10% test at each fold so ten training sets are used here,\n")
        ftrain.write("Interation %d ,\n" %(metcount+1))
ftrain.close()

with open(test_name,'w') as fpred:
        fpred.write("Code originally created by James L. McDonagh 2016 for use in predicting sublimation thermodynamics.\n")
        fpred.write("This file contains the prediction information for all three models (Random Forest, Support Vector Regression and Partial Least Squares).\n")
        fpred.write("Predictions are made over a ten fold cross validation hence training on 90% test on 10%. The final prediction are return iteratively over this ten fold cros validation once,\n")
        fpred.write("optimised parameters are located via a grid search at each fold,\n")
        fpred.write("Interation %d ,\n" %(metcount+1))
fpred.close()

with open(fi_name,'w') as ffeatimp:
        ffeatimp.write("Code originally created by James L. McDonagh 2016 for use in predicting sublimation thermodynamics,\n")
        ffeatimp.write("This file contains the feature importance information for the Random Forest model,\n")
        ffeatimp.write("Interation %d ,\n" %(metcount+1))
ffeatimp.close()

# Begin the K-fold cross validation over ten folds
kf = KFold(datax, n_folds=10, shuffle=True, random_state=0)
print "------------------- Begining Ten Fold Cross Validation -------------------"
for train, test in kf:
	XTrain, XTest, yTrain, yTest = X[train], X[test], y[train], y[test]
	ytestdim = yTest.shape[0]
        print("The test set values are : ")
        i = 0
        with open (train_name, 'a') as ftrain:
                if ytestdim%5 == 0:
                        while i < ytestdim:
                                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2),'\t', round(yTest[i+3],2),'\t', round(yTest[i+4],2)
                                ftrain.write(str(round(yTest[i],2))+','+ str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+','+str(round(yTest[i+3],2))+','+str(round(yTest[i+4],2))+',\n')
                                i += 5
                elif ytestdim%4 == 0:
                        while i < ytestdim:
                                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2),'\t', round(yTest[i+3],2)
                                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+','+str(round(yTest[i+3],2))+',\n')
                                i += 4
                elif ytestdim%3 == 0 :
                        while i < ytestdim :
                                print round(yTest[i],2),'\t', round(yTest[i+1],2),'\t', round(yTest[i+2],2)
                                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+','+str(round(yTest[i+2],2))+',\n')
                                i += 3
                elif ytestdim%2 == 0 :
                        while i < ytestdim :
                                print round(yTest[i],2), '\t', round(yTest[i+1],2)
                                ftrain.write(str(round(yTest[i],2))+','+str(round(yTest[i+1],2))+',\n')
                                i += 2
                        else :
                                while i< ytestdim :
                                        print round(yTest[i],2)
                                        ftrain.write(str(round(yTest[i],2))+',\n')
                                        i += 1
        ftrain.close()

        print "\n"
        # random forest grid search parameters
	print "------------------- Begining Random Forest Grid Search -------------------"
        rfparamgrid = {"n_estimators": [10, 50, 100, 500], "max_features": ["auto", "sqrt", "log2"], "max_depth": [5,7]}
        rf = RandomForestRegressor(random_state=0,n_jobs=2)
        RfGridSearch = GridSearchCV(rf,param_grid=rfparamgrid,scoring='mean_squared_error',cv=10)
        start = time()
        RfGridSearch.fit(XTrain,yTrain)

        # Get best random forest parameters
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings" %(time() - start,len(RfGridSearch.grid_scores_)))
        RFtime = time() - start,len(RfGridSearch.grid_scores_)
        #print(RfGridSearch.grid_scores_)
        print("n_estimators = %d " % RfGridSearch.best_params_['n_estimators'])
        ne = RfGridSearch.best_params_['n_estimators']
        print("max_features = %s " % RfGridSearch.best_params_['max_features'])
        mf = RfGridSearch.best_params_['max_features']
        print("max_depth = %d " % RfGridSearch.best_params_['max_depth'])
        md = RfGridSearch.best_params_['max_depth']
        with open (train_name, 'a') as ftrain:
                ftrain.write("Random Forest")
                ftrain.write("RF search time, %s ,\n" % (str(RFtime)))
                ftrain.write("Number of Trees, %s ,\n" % str(ne))
                ftrain.write("Number of feature at split, %s ,\n" % str(mf))
                ftrain.write("Max depth of tree, %s ,\n" % str(md))
        ftrain.close()
                             
	# support vector regression grid search paramters
	print "------------------- Begining Support Vector Regrssion Grid Search -------------------"
	svrparamgrid = {"C": [0.25,0.5,1.0,5,10,50,100,500,1000], "epsilon": [0.5, 1, 2, 3,5, 10], "gamma": ["auto",1,5,10]}
	svmr = SVR(kernel = 'rbf')
	SvmrGridSearch = GridSearchCV(svmr, param_grid=svrparamgrid, scoring="mean_squared_error",cv=10)
	start = time()
	SvmrGridSearch.fit(XTrain,yTrain)

	# Get best support vector regression parameters
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings" %(time() - start,len(SvmrGridSearch.grid_scores_)))
        SVRtime = time() - start,len(SvmrGridSearch.grid_scores_)
        print("Capacity (C) = %d " % SvmrGridSearch.best_params_['C'])
        capacity = SvmrGridSearch.best_params_['C']
        print("Epsilon = %s " % SvmrGridSearch.best_params_['epsilon'])
        ep = SvmrGridSearch.best_params_['epsilon']
	print("Gamma (kernel coefficent) = %s " % SvmrGridSearch.best_params_['gamma'])
	ga = SvmrGridSearch.best_params_['gamma']
        with open (train_name, 'a') as ftrain:
                ftrain.write("Support Vector Regression")
                ftrain.write("SVR search time, %s ,\n" % (str(SVRtime)))
                ftrain.write("Default Radial Basis Kernel used,\n")
                ftrain.write("Capacity (C), %s ,\n" % (str(capacity)))
                ftrain.write("Epsilon (extent of the corridor of zero penalty from the loss function), %s ,\n" % (str(ep)))
                ftrain.write("Kernel Coefficent, %s ,\n" % (str(ga)))
        ftrain.close()

        # partial least squares grid search paramters
	print "------------------- Begining Partial Least Squares Grid Search -------------------"
	plsrparamgrid = {"n_components": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
	plsr = PLSRegression()
	PlsrGridSearch = GridSearchCV(plsr, param_grid=plsrparamgrid, scoring="mean_squared_error",cv=10)
	start = time()
	PlsrGridSearch.fit(XTrain,yTrain)

	# Get best partial least squares parameters
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings" %(time() - start,len(PlsrGridSearch.grid_scores_)))
        PLStime = time() - start,len(PlsrGridSearch.grid_scores_)
        print("Number of components = %d " % PlsrGridSearch.best_params_['n_components'])
        nc = PlsrGridSearch.best_params_['n_components']
        with open (train_name, 'a') as ftrain:
                ftrain.write("Partial Least Squares")
                ftrain.write("PLS search time, %s ,\n" % (str(PLStime)))
                ftrain.write("Number of Components, %s ,\n" % (str(nc)))
        ftrain.close()

        # Train random forest and predict with optimised parameters
        print("\n\n------------------- Starting opitimised RF training -------------------")
        optRF = RandomForestRegressor(n_estimators = ne, max_features = mf, max_depth = md, random_state=0)
        optRF.fit(XTrain, yTrain)       # Train the model
        RFfeatimp = optRF.feature_importances_
        indices = np.argsort(RFfeatimp)[::-1]
        print("Training R2 = %5.2f" % optRF.score(XTrain,yTrain))
        print("Starting optimised RF prediction")
        RFpreds = optRF.predict(XTest)
        print("The predicted values now follow :")
        RFpredsdim = RFpreds.shape[0]
        i = 0
        if RFpredsdim%5 == 0:
                while i < RFpredsdim:
                        print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2),'\t', round(RFpreds[i+3],2),'\t', round(RFpreds[i+4],2)
                        i += 5
        elif RFpredsdim%4 == 0:
                while i < RFpredsdim:
                        print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2),'\t', round(RFpreds[i+3],2)
                        i += 4
        elif RFpredsdim%3 == 0 :
                while i < RFpredsdim :
                        print round(RFpreds[i],2),'\t', round(RFpreds[i+1],2),'\t', round(RFpreds[i+2],2)
                        i += 3
        elif RFpredsdim%2 == 0 :
                while i < RFpredsdim :
                        print round(RFpreds[i],2), '\t', round(RFpreds[i+1],2)
                        i += 2
        else :
                while i< RFpredsdim :
                        print round(RFpreds[i],2)
                        i += 1
        print "\n"
        RFr2.append(optRF.score(XTest, yTest))
        RFmse.append( metrics.mean_squared_error(yTest,RFpreds))
        RFrmse.append(math.sqrt(RFmse[metcount]))
        print ("Random Forest prediction statistics for fold %d are; MSE = %5.2f RMSE = %5.2f R2 = %5.2f\n\n" % (metcount+1, RFmse[metcount], RFrmse[metcount],RFr2[metcount]))
        with open(train_name,'a') as ftrain :
                ftrain.write("Random Forest prediction statistics for fold %d are, MSE =, %5.2f, RMSE =, %5.2f, R2 =, %5.2f,\n\n" % (metcount+1, RFmse[metcount], RFrmse[metcount],RFr2[metcount]))
        ftrain.close()
        
        with open(fi_name,'a') as ffeatimp:
                ffeatimp.write("Feature importance rankings from random forest,\n")
                for i in range(RFfeatimp.shape[0]) :
                        ffeatimp.write("%d. , feature %d , %s,  (%f),\n" % (i + 1, indices[i], npheader[indices[i]], RFfeatimp[indices[i]]))
        ffeatimp.close()

        # Train Support Vector regression model and predict with optimised parameters
        print("\n\n------------------- Starting opitimised SVR training -------------------")
        optSVR = SVR(C = capacity, epsilon = ep, gamma = ga)
        optSVR.fit(XTrain, yTrain)       # Train the model
        print("Training R2 = %5.2f" % optSVR.score(XTrain,yTrain))
        print("Starting optimised SVR prediction")
        SVRpreds = optSVR.predict(XTest)
        print("The predicted values now follow :")
        SVRpredsdim = SVRpreds.shape[0]
        i = 0
        if SVRpredsdim%5 == 0:
                while i < SVRpredsdim:
                        print round(SVRpreds[i],2),'\t', round(SVRpreds[i+1],2),'\t', round(SVRpreds[i+2],2),'\t', round(SVRpreds[i+3],2),'\t', round(SVRpreds[i+4],2)
                        i += 5
        elif SVRpredsdim%4 == 0:
                while i < SVRpredsdim:
                        print round(SVRpreds[i],2),'\t', round(SVRpreds[i+1],2),'\t', round(SVRpreds[i+2],2),'\t', round(SVRpreds[i+3],2)
                        i += 4
        elif SVRpredsdim%3 == 0 :
                while i < SVRpredsdim :
                        print round(SVRpreds[i],2),'\t', round(SVRpreds[i+1],2),'\t', round(SVRpreds[i+2],2)
                        i += 3
        elif SVRpredsdim%2 == 0 :
                while i < SVRpredsdim :
                        print round(SVRpreds[i],2), '\t', round(SVRpreds[i+1],2)
                        i += 2
        else :
                while i< SVRpredsdim :
                        print round(SVRpreds[i],2)
                        i += 1
        print "\n"
        SVRr2.append(optSVR.score(XTest, yTest))
        SVRmse.append( metrics.mean_squared_error(yTest,SVRpreds))
        SVRrmse.append(math.sqrt(SVRmse[metcount]))
        print ("Support Vector Regression prediction statistics for fold %d are; MSE = %5.2f RMSE = %5.2f R2 = %5.2f\n\n" % (metcount+1, SVRmse[metcount], SVRrmse[metcount],SVRr2[metcount]))
        with open(train_name,'a') as ftrain :
                ftrain.write("Support Vector Regression prediction statistics for fold %d are, MSE =, %5.2f, RMSE =, %5.2f, R2 =, %5.2f,\n\n" % (metcount+1, SVRmse[metcount], SVRrmse[metcount],SVRr2[metcount]))
        ftrain.close()

        # Train partial least squares and predict with optimised parameters
        print("\n\n------------------- Starting opitimised PLS training -------------------")
        optPLS = PLSRegression(n_components = nc)
        optPLS.fit(XTrain, yTrain)       # Train the model
        print("Training R2 = %5.2f" % optPLS.score(XTrain,yTrain))
        print("Starting optimised PLS prediction")
        PLSpreds = optPLS.predict(XTest)
        print("The predicted values now follow :")
        PLSpredsdim = PLSpreds.shape[0]
        i = 0
        if PLSpredsdim%5 == 0:
                while i < PLSpredsdim:
                        print round(PLSpreds[i],2),'\t', round(PLSpreds[i+1],2),'\t', round(PLSpreds[i+2],2),'\t', round(PLSpreds[i+3],2),'\t', round(PLSpreds[i+4],2)
                        i += 5
        elif PLSpredsdim%4 == 0:
                while i < PLSpredsdim:
                        print round(PLSpreds[i],2),'\t', round(PLSpreds[i+1],2),'\t', round(PLSpreds[i+2],2),'\t', round(PLSpreds[i+3],2)
                        i += 4
        elif PLSpredsdim%3 == 0 :
                while i < PLSpredsdim :
                        print round(PLSpreds[i],2),'\t', round(PLSpreds[i+1],2),'\t', round(PLSpreds[i+2],2)
                        i += 3
        elif PLSpredsdim%2 == 0 :
                while i < PLSpredsdim :
                        print round(PLSpreds[i],2), '\t', round(PLSpreds[i+1],2)
                        i += 2
        else :
                while i< PLSpredsdim :
                        print round(PLSpreds[i],2)
                        i += 1
        print "\n"
        PLSr2.append(optPLS.score(XTest, yTest))
        PLSmse.append(metrics.mean_squared_error(yTest,PLSpreds))
        PLSrmse.append(math.sqrt(PLSmse[metcount]))
        print ("Partial Least Squares prediction statistics for fold %d are; MSE = %5.2f RMSE = %5.2f, R2 = %5.2f\n\n" % (metcount+1, PLSmse[metcount], PLSrmse[metcount],PLSr2[metcount]))
        with open(train_name,'a') as ftrain :
                 ftrain.write("Partial Least Squares prediction statistics for fold %d are, MSE =, %5.2f, RMSE =, %5.2f, R2 =, %5.2f,\n\n" % (metcount+1, PLSmse[metcount], PLSrmse[metcount],PLSr2[metcount]))
        ftrain.close()
        
        # Store prediction in original order of data (itest) whilst following through the current test set order (j)
	metcount += 1
        with open(train_name,'a') as ftrain :
                ftrain.write("Fold %d, \n" %(metcount))
        
	print "------------------- Next Fold %d -------------------" %(metcount+1)
	j = 0
	for itest in test :
		RFpredictions.append(RFpreds[j])
                SVRpredictions.append(SVRpreds[j])
                PLSpredictions.append(float(PLSpreds[j]))
		j += 1

with open(test_name,'a') as fpred :
        lennames = names.shape[0]
        lenpredictions = len(RFpredictions)
        lentrue = y.shape[0]
        if lennames == lenpredictions == lentrue :
                fpred.write("Names/Label,, Prediction Random Forest,, Prediction Support Vector Regression ,, Prediction Partial Least Squares ,, True Value,\n") 
                for i in range(0,lennames) :
                        fpred.write(str(names[i])+",,"+str(RFpredictions[i])+",,"+str(SVRpredictions[i])+",,"+str(PLSpredictions[i])+",,"+str(y[i])+",\n")
        else :
                print "ERROR - names, prediction and true value array size mismatch. Dumping arrays for manual inspection in predictions.csv"
                fpred.write("ERROR - names, prediction and true value array size mismatch. Dumping arrays for manual inspection in predictions.csv\n")
                fpred.write("Array printed in the order names/Labels, predictions RF and true values\n")
                fpred.write(names+"\n")
                fpred.write(RFpredictions+"\n")
                fpred.write(y+"\n")
                
print "Final averaged Random Forest metrics : "
RFamse  = sum(RFmse)/10
RFmse_sd = np.std(RFmse)
RFarmse = sum(RFrmse)/10
RFrmse_sd = np.std(RFrmse)
RFslope, RFintercept, RFr_value, RFp_value, RFstd_err = scipy.stats.linregress(RFpredictions, y)
RFR2 = RFr_value**2
print "Average Mean Squared Error = ", RFamse, " +/- ", RFmse_sd 
print "Average Root Mean Squared Error = ", RFarmse, " +/- ", RFrmse_sd
print "R2 Final prediction against True values = ", RFR2

print "Final averaged Support Vector Regression metrics : "
SVMRamse  = sum(SVRmse)/10
SVMRmse_sd = np.std(SVRmse)
SVMRarmse = sum(SVRrmse)/10
SVMRrmse_sd = np.std(SVRrmse)
SVMRslope, SVMRintercept, SVMRr_value, SVMRp_value, SVMRstd_err = scipy.stats.linregress(SVRpredictions, y)
SVMRR2 = SVMRr_value**2
print "Average Mean Squared Error = ", SVMRamse
print "Average Root Mean Squared Error = ", SVMRarmse
print "R2 Final prediction against True values = ", SVMRR2

print "Final averaged Partial Least Squares metrics : "
PLSamse  = sum(PLSmse)/10
PLSmse_sd = np.std(PLSmse)
PLSarmse = sum(PLSrmse)/10
PLSrmse_sd = np.std(PLSrmse)
PLSslope, PLSintercept, PLSr_value, PLSp_value, PLSstd_err = scipy.stats.linregress(PLSpredictions, y)
PLSR2 = PLSr_value**2
print "Average Mean Squared Error = ", sum(PLSmse)/10
print "Average Root Mean Squared Error = ", sum(PLSrmse)/10
print "R2 Final prediction against True values = ", PLSR2

with open(test_name,'a') as fpred :
        fpred.write("\n")
        fpred.write("FINAL PREDICTION STATISTICS,\n")
        fpred.write("Random Forest average MSE, %s, +/-, %s,\n" %(str(RFamse), str(RFmse_sd)))
        fpred.write("Random Forest average RMSE, %s, +/-, %s,\n" %(RFarmse, RFrmse_sd))
        fpred.write("Random Forest R2, %s,\n" %(str(RFR2)))
        fpred.write("Support Vector Machine Regression average MSE, %s, +/-, %s,\n" %(str(SVMRamse),str(SVMRmse_sd)))
        fpred.write("Support Vector Machine Regression average RMSE, %s, +/-, %s,\n" %(str(SVMRarmse),str(SVMRrmse_sd)))
        fpred.write("Support Vector Machine Regression R2, %s ,\n" %(str(SVMRR2)))
        fpred.write("Partial Least Squares average MSE, %s, +/-, %s,\n" %(str(PLSamse),str(PLSmse_sd)))
        fpred.write("Partial Least Squares average RMSE, %s, +/-, %s,\n" %(str(PLSarmse),str(PLSrmse_sd)))
        fpred.write("Partial Least Squares R2, %s ,\n" %(str(PLSR2)))

