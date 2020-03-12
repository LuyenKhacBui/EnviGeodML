# recursive multi-step forecast with linear algorithms
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import sys, h5py, time, os
import numpy as np
from Utils import prgrsTime

# split a univariate dataset into train/test sets
def split_dataset(data,sttrain,fntrain,fntest,blocklen):
	# split into years    
	train = data[sttrain:fntrain+1]
	# restructure into windows of weekly data: Divide data into 12-month blocks
    # after below lines, train, test will be of [years, months, features]
	train = array(split(train, len(train)/blocklen))
	if not np.isnan(fntest):
		test = data[fntrain+1:fntest+1]
		test = array(split(test, len(test)/blocklen))
	else:
		test=[]	
	return train, test

# evaluate one or more monthly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each month
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	r2 = r2_score(actual.flatten(),predicted.flatten())
	return score, scores, r2

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# prepare a list of ml models
def get_models(models=dict()):
	# linear models	
	#models['lr'] = LinearRegression()
	'''#models['lasso'] = Lasso()		# Bad results
	models['ridge'] = Ridge()
	#models['en'] = ElasticNet()	# Bad results
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	#models['llars'] = LassoLars()	# Bad results
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ransac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	#print('Defined %d models' % len(models))'''
	#models['forestreg'] = RandomForestRegressor()
	#models['treereg'] = DecisionTreeRegressor()
	models['SVMreg'] = LinearSVR()
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# make a recursive multi-step forecast
def forecast(model, input_x, n_input):
	''' Fn used to forecast of univariate, e.g., from the last year to the next year
	input:
		- model:   model under consideration in dictionary
		- input_x: [Jan Feb ... Dec]	=>	of the last year
		- n_input: number of [linear] paras also the length will be predicted
	output:
		- yhat_sequence: list of forecasted values for the next year
				   [Jan Feb ... Dec]	=>	of the next year
	'''	
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(n_input):
		# prepare the input data
		X = array(input_data[-n_input:]).reshape(1, n_input)
		# make a one-step forecast
		yhat = model.predict(X)[0]
		# add to the result
		yhat_sequence.append(yhat)
		# add the prediction to the input
		input_data.append(yhat)
	return yhat_sequence

# convert windows of yearly multivariate data into a series of, e.g., GRACE
def to_series(data,datidx): # a list of studied data with len: years*12
	# extract just the studied data (GRACE) from each year
	series = [mnt[:,datidx] for mnt in data]
	# flatten into a single series
	series = array(series).flatten()	
	return series

# convert history into inputs and outputs
def to_supervised(history, n_input, datidx):
	# convert history to a univariate series
	data = to_series(history, datidx) # an array of studied data (GRACE) with size (yrs*12,)
	X, y = list(), list()
	ix_start = 0
	# step over the entire history one time step at a time
	for i in range(len(data)):
		# define the end of the input sequence
		ix_end = ix_start + n_input
		# ensure we have enough data for this instance
		if ix_end < len(data):
			X.append(data[ix_start:ix_end])
			y.append(data[ix_end])
		# move along one time step
		ix_start += 1
	return array(X), array(y)

# fit a model and make a forecast
def sklearn_predict(model, history, n_input, datidx):
	''' Fn used to predict from multivariate. Note this does NOT work with multivariate.
											  Instead, it simply extracts one variate from multivariate to work.
	Input:
		- model:   model under studied in dictionary format
		- history: [yr1[1,2...], yr2[1,2,...], ...]. yri[j] = [Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec]
		- n_input: number of [linear] paras. = 12 in cased of monthly, 7 in case of daily
		- datidx:  index of studied data, 0 for our case of GRACE if [GRACE, GLDAS, TRMM, ...]
	Implementation:
		- train/test split: train = [[Jan Feb ... Dec],[Feb Apr ... Jan], [Apr May ... Feb], ...]
							test  = [            [Jan],             [Feb],            [Mar], ...]
		- make a pipeline:  1. standardization, 2. normalization, 3. model
		- fit based on train/test
		- forecast: use the last year in train (i.e., the last element of the list) [Jan Feb ... Dec]
													to forecast for the next year	[Jan Feb ... Dec]
	Output:
		- yhat_sequence: a list of forecasted values for the next year [Jan Feb ... Dec]
	'''
	# prepare data
	train_x, train_y = to_supervised(history, n_input, datidx)
	# make pipeline
	pipeline = make_pipeline(model)
	# fit the model
	pipeline.fit(train_x, train_y)
	# predict the week, recursively
	yhat_sequence = forecast(pipeline, train_x[-1, :], n_input) # Luyen: make sure to change paras in this
	return yhat_sequence

# evaluate a single model
def evaluate_model(model, train, test, n_input, datidx):
	''' Fn used to evaluate a model in univariate time series forecasting
	Input:
		model: model under evaluation in dictinary format
		train: train data set
	Output:
	'''
	# history is a list of monthly data. train is an array [yrs,mnts,features]=(11,12,3)
	# this simply converts from an array to a list, each element of the list is a year with [months,features]=[12,3]
	history = [x for x in train] 
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)): # test is [yrs,mnts,featues]=[3,12,3]
		# predict for the next year from the list of previous years
		yhat_sequence = sklearn_predict(model, history, n_input, datidx)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions) # = the size of test
	# evaluate predictions months for each year
	score, scores, r2 = evaluate_forecasts(test[:, :, datidx], predictions)
	return score, scores, r2, predictions

# predict by a single model
def predict_model(model, train, n_input, datidx, n_year):
	''' Fn used to predict for coming months by a model in univariate time series forecasting
	Input:
		model:   model under evaluation in dictinary format
		train:   train data set in (year,month,features)=(year,12,3)
		n_input: number of [linear] paras, also the length will be predicted. 12 in case of monthly data
		datidx:  index of studied data. 0 in this case of GRACE
		n_year:  number of year will be predicted. The predicted values of previous year will be appended to train to predict current year
	Output:
		predictions: predicted values in (n_year,month)=(n_year,12)
	'''
	# history is a list of monthly data. train is an array [yrs,mnts,features],e.g., (11,12,3)
	# this simply converts from an array to a list, each element of the list is a year with [months,features]=[12,3]
	history = [x for x in train] 
	# walk-forward prediction over each year
	predictions = list()
	for i in range(n_year):
		# predict for the next year from the list of previous years
		yhat_sequence = sklearn_predict(model, history, n_input, datidx)
		# store the predictions
		predictions.append(yhat_sequence)
		# add predicted values to history for predicting the next week
		history.append(np.hstack((array(yhat_sequence).reshape(len(yhat_sequence),1),np.full((train.shape[1],train.shape[2]-1),np.nan))))
	predictions = array(predictions) # = the size of test
	return predictions

def h5read(h5file, h5dsetin, mat2py = False):
	# Read data sets from a .h5 file
	h5data = h5py.File(h5file, 'r')
	outlst = []
	for itemin in h5dsetin:
		arr = h5data[itemin].value
		if mat2py == True:
			arr = np.transpose(arr, (2,1,0))
		outlst.append(arr)
	return outlst

def nanfill(arr, method='linear'):
	'''	Fill NaNs with linear interpolation from nearest values
		If there are non NaN values before and after a NaN location, they will be used for linear interpolation
		If there is not non NaN value beforer or after a NaN location, the nearest non NaN value will be assigned
			to this NaN location
	'''
	if method=='linear':
		nans = np.isnan(arr) # a list of True/False. True if NaN and False if non NaN
		idxs = lambda z: z.nonzero()[0]	# returns the indices of a value. Used below for indices of NaN and non NaN	
		arr[nans] = np.interp(idxs(nans), idxs(~nans), arr[~nans])
	return arr
	
# Main program from here
print('Load monthly data')
[grace,gldas,trmm] = h5read('DataInMLbyMatlab.h5',['GRACE_TWS','GLDAS_SM_ANOM','TRMM_PRECIP_ANOM'],mat2py=True)
[fulltime] = h5read(h5file = 'DataInMLbyMatlab.h5', h5dsetin = ['fulltime'], mat2py=False)
timestr = [str(item)[1:9] for item in fulltime] # a list of string in yyyymmdd
#timenum = [float(item[0:4])+float(item[4:6])/12 for item in timestr]
[nlen, nwid, ntim] = grace.shape
dataset = np.empty((ntim, 3))
missmnths,blclen = 22+29,12 # GRACE has 22 missing months and 29 months from 2017.07 to 2019.11. They were all filled-in with NaNs in Matlab
datidx = 0 # research data. 0 = GRACE, 1=GLDAS SM, 2=TRMM precipitation
# prepare the models to evaluate
models = get_models()
# Split data to train (2003.01-2013.12) and test(2014.01-2016.12) used for validation
sttrain,fntrain,fntest = timestr.index('20020601'),timestr.index('20140501'),timestr.index('20170501')
modeltest = np.full((nlen,nwid,fntest-fntrain),np.nan)
# split data to train (2003.01-2016.12) used for prediction to the next 2 years
stpred,fnpred,predyr = timestr.index('20020601'),timestr.index('20170501'),1 # 2=predict to the next 2 years
modelpred = np.full((nlen,nwid,predyr*blclen),np.nan)
# evaluate then do the prediction for each model
for no, [name, model] in enumerate(models.items()):	
	print("Implement the Recursive Multi-step Forecast on model no: %d / %d %s" %(no+1, len(models), type(model).__name__))	
	score, scores, r2 = np.full((nlen,nwid),np.nan), np.full((nlen,nwid,blclen),np.nan), np.full((nlen,nwid),np.nan)
	tic = time.time()
	intvlprct = max(round(float(nlen)/10), 1)
	for row in range(nlen):
		for col in range(nwid):
			if np.count_nonzero(np.isnan(grace[row, col, :])==True) > missmnths:
				pass
			else:
				dataset[0:ntim,0],dataset[0:ntim,1],dataset[0:ntim,2] = grace[row, col, :],gldas[row, col, :],trmm[row, col, :]			
				dataset[0:ntim,datidx] = nanfill(dataset[0:ntim,0],'linear') # Fill NaNs by linear interpolation												
				# for evaluation
				train, test = split_dataset(dataset,sttrain,fntrain,fntest,blclen) # make sure to change paras in this fn if needed							
				n_input = blclen
				# evaluate and get scores					
				# score:  rmse computed from all predicted values
				# scores: rmse computed for each month. rmse of [Jan Feb ... Dec]
				score[row,col],scores[row,col,:],r2[row,col],predictions = evaluate_model(model, train, test, n_input, datidx) # 0 is the index of studied data, GRACE in this
				modeltest[row,col,:] = predictions.flatten()
				# for prediction
				train, test = split_dataset(dataset,stpred,fnpred,np.nan,blclen) # test will be [] and not be used here
				predictions = predict_model(model, train, n_input, datidx, predyr)
				modelpred[row,col,:] = predictions.flatten()
		if (row + 1) % intvlprct == 0 or (row + 1) == nlen:
			prgrs_prct = float(row + 1) / nlen * 100
			prgrs_runtime = time.time() - tic
			prgrs_str = '\t\tRow: ' + str(row + 1) + ' / ' + str(nlen) + ' ---> ' + str(int(round(prgrs_prct))) + ' % '
			f = prgrsTime(prgrs_str, prgrs_prct, prgrs_runtime)
	# Save to .h5 file
	h5ofile = type(model).__name__ + '/' + type(model).__name__ + '.h5'
	if not os.path.exists(type(model).__name__):
		print('\tMaking dir %s' %(type(model).__name__))
		os.mkdir(type(model).__name__)
	if os.path.exists(h5ofile):
		print('\tDeleting previous %s'%(h5ofile))
		os.remove(h5ofile)		
	print('\tSave results to .h5 file: %s' %(h5ofile))
	f = h5py.File(h5ofile,"w")
	f.attrs['help'] = 'Apply Recursive Multi-step to forecast gaps in GRACE TWS'
	dset = f.create_dataset('score',data=score)
	dset.attrs['help'] = 'Pixels score computed from all predicted values of all months over the test period '+timestr[fntrain+1]+'-'+timestr[fntest]+' predicted from observations '+timestr[sttrain]+'-'+timestr[fntrain]
	dset = f.create_dataset('r2',data=r2)
	dset.attrs['help'] = 'Pixels R-squared computed from all predicted values of all months over the test period '+timestr[fntrain+1]+'-'+timestr[fntest]+' predicted from observations '+timestr[sttrain]+'-'+timestr[fntrain]
	dset = f.create_dataset('scores',data=scores)
	dset.attrs['help'] = 'Pixels score computed from all predicted values for each month over the test period '+timestr[fntrain+1]+'-'+timestr[fntest]+' predicted from observations '+timestr[sttrain]+'-'+timestr[fntrain]
	dset = f.create_dataset('fulltime',data=fulltime)
	dset.attrs['help'] = 'Time of full data sets in YYYYMMDD format'
	dset = f.create_dataset('modeltest',data=modeltest)
	dset.attrs['help'] = 'Predicted values over the test period '+timestr[fntrain+1]+'-'+timestr[fntest]+' predicted from observations '+timestr[sttrain]+'-'+timestr[fntrain]
	dset = f.create_dataset('modelpred',data=modelpred)
	dset.attrs['help'] = 'Predicted values over the prediction period '+timestr[fnpred+1]+'-'+timestr[fnpred+predyr*blclen]+' predicted from observation '+timestr[stpred]+'-'+timestr[fnpred]
	f.close()

	'''			trainseries = train[:,:,datidx].flatten()
				testseries  = test[:,:,datidx].flatten()
				pred = predictions.flatten()
				#pyplot.plot(np.array(timenum[sttrain:fntrain]),trainseries,marker='o')
				#pyplot.plot(np.array(timenum[fntrain:]), testseries, marker='o')
				#pyplot.plot(np.array(timenum[fntrain:]), pred, marker='o')
				pyplot.show()
				a=1
				# summarize scores
				summarize_scores(name, score, scores)
				# plot scores
				pyplot.plot(mnts, scores, marker='o', label=name)
			# show plot
			pyplot.legend()
			pyplot.show()
			'''