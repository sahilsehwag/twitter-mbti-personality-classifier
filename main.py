# ================ IMPORTS ======================
import sys
import os
import re
import string
import path

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import tweepy
from colorama import *

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from nltk.stem   import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics                 import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing           import LabelEncoder, OneHotEncoder
from sklearn.model_selection         import train_test_split, cross_val_score
from sklearn.externals               import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble    import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm         import SVC
# ================ IMPORTS ======================



# ================= HELPERS =====================
#colorama initialization
def printl(msg):
	sys.stdout.write("\r" + str(msg))
	sys.stdout.flush()
# ================= HELPERS =====================



# ================= DATASET =====================
def load_dataset(filename='personality-test.csv'):
	df = pd.read_csv(filename)
	df['ie'] = df.type
	df['ns'] = df.type
	df['ft'] = df.type
	df['pj'] = df.type

	for i, t in enumerate(df.type):
		if 'I' in t:
			df.ie[i] = 'I'
		elif 'E' in t:
			df.ie[i] = 'E'

		if 'N' in t:
			df.ns[i] = 'N'
		elif 'S' in t:
			df.ns[i] = 'S'

		if 'F' in t:
			df.ft[i] = 'F'
		elif 'T' in t:
			df.ft[i] = 'T'

		if 'P' in t:
			df.pj[i] = 'P'
		elif 'J' in t:
			df.pj[i] = 'J'

	return df
def create_train_test_sets(dataset, X):
	#getting target columns for each personality characterstic pair
	ys	= [
		dataset.type.values,
		dataset.ie.values,
		dataset.ns.values,
		dataset.ft.values,
		dataset.pj.values,
	]

	#splitting dataset into training and testing dataset
	xTrains, yTrains, xTests, yTests = [],[],[],[]
	for y in ys:
		xTrain, xTest, yTrain, yTest = train_test_split(X, y)
		xTrains.append(xTrain)
		yTrains.append(yTrain)
		xTests.append(xTest)
		yTests.append(yTest)

	return ys, xTrains, yTrains, xTests, yTests
# ================= DATASET =====================



# ============== PREPROCESSING ==================
#regular expressions for tokenization
regexes = [
	#punctuation
	r'(?:(\w+)\'s)',

	r'(?:\s(\w+)\.+\s)',
	r'(?:\s(\w+),+\s)',
	r'(?:\s(\w+)\?+\s)',
	r'(?:\s(\w+)!+\s)',

	r'(?:\'+(\w+)\'+)',
	r'(?:"+(\w+)"+)',
	r'(?:\[+(\w+)\]+)',
	r'(?:{+(\w+)}+)',
	r'(?:\(+(\w+))',
	r'(?:(\w+)\)+)',

	#words containing numbers & special characters & punctuation
	r'(?:(?:(?:[a-zA-Z])*(?:[0-9!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~])+(?:[a-zA-Z])*)+)',

	#pure words
	r'([a-zA-Z]+)',
]

#compiling regular expression
regex = re.compile(r'(?:'+'|'.join(regexes)+')', re.VERBOSE | re.IGNORECASE)


def preprocess(documents):
	lemmatizer = WordNetLemmatizer()
	stemmer = PorterStemmer()

	#fetching list of stopwords
	punctuation = list(string.punctuation)
	swords = stopwords.words('english') + ['amp'] + ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january', 'feburary', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',  'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',	'jan', 'feb', 'mar', 'apr', 'may', 'jun' 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'tommorow', 'today', 'yesterday'] + ['mr', 'mrs']

	processed_documents = []
	for i,document in enumerate(documents):
		printl('{0}/{1}'.format(i+1, len(documents)))

		#tokenization
		tokens = regex.findall(document)

		#skipping useless tokens
		t_regex = re.compile(r"[^a-zA-Z]")
		document = []

		for token in tokens:
			token = np.array(token)
			token = np.unique(token[token != ''])

			if len(token) > 0:
				token = token[0].lower()
			else:
				continue

			if re.search(t_regex, token) == None and token not in swords:
				token = lemmatizer.lemmatize(token)
				document.append(token)

		document = ' '.join(document)

		#skipping
		if len(document) > 0:
			processed_documents.append(document)

	print()
	return np.array(processed_documents)


# ============== PREPROCESSING ==================



# =============== MODELING ======================
def get_models(dataset='personality-test.csv', filename='models.pkl', overwrite=False, verbose=False):
	if os.path.isfile(filename) and not overwrite:
		if verbose:
			print(Fore.BLACK + '✓ pickled models found')
			print(Fore.GREEN + '✓ models loaded')
		models = joblib.load(filename)
	else:
		if verbose:
			print(Fore.RED + '✖ models not found')
			print(Fore.BLACK + 'training new models...')

		#loading dataset
		dataset = load_dataset(dataset)

		if verbose:
			print(Fore.GREEN + '✓ dataset loaded')
			print(Fore.BLACK + 'processing dataset...')

		#creating dataset
		cv = CountVectorizer().fit(dataset.posts.values)
		X  = cv.transform(dataset.posts.values)

		#creating training and test splits for all target types
		ys, xTrains, yTrains, xTests, yTests = create_train_test_sets(dataset, X)

		#training our models
		models = []
		for i in range(len(ys)):
			if verbose:
				printl(Fore.BLACK + 'training model {0}/{1}...'.format(i+1, len(ys)))
			model = MultinomialNB().fit(xTrains[i], yTrains[i])
			models.append(model)
		print()

		if verbose:
			print(Fore.GREEN + '✓ all models trained')

		#getting cross-validation scores
		accuracies = []
		for i,model in enumerate(models):
			if verbose:
				printl(Fore.BLACK + 'computing cross validation score for model {0}/{1}'.format(i+1, len(models)))
			accuracies.append(cross_val_score(
				estimator=model, 
				cv=10, 
				X=X, y=ys[i], 
				n_jobs=-1))
		print()

		if verbose:
			print(Fore.GREEN + '✓ cross validation done')

		models = {
			'models'     : models,
			'accuracies' : accuracies,
			'cv'         : cv }

		#pickling models
		joblib.dump(models, filename)

		if verbose:
			print(Fore.GREEN + '✓ models pickled to disk({0})'.format(filename))

	return models
# =============== MODELING ======================



# ================ TWITTER ======================
def get_user_tweets(api, username, count=200):
	tweets = api.user_timeline(username, count=count)
	texts = [tweet.text for tweet in tweets]

	return texts
# ================ TWITTER ======================



# ============= CLI INTERFACE ===================
# ============= CLI INTERFACE ===================



# ============= BOOTSTRAPPING ===================
if __name__ == '__main__':
	#colorama initialization
	init(autoreset=True)

	verbose = True

	#getting cli arguments
	args = sys.argv
	try:
		username = args[1]
	except Exception:
		username = None


	#getting models if already serialized
	#else creating, training and pickling our models
	models     = get_models(verbose=verbose)
	accuracies = models['accuracies']
	cv         = models['cv']
	models     = models['models']


	#twitter authentication
	CONSUMER_KEY        = os.environ['TWITTER_CONSUMER_KEY']
	CONSUMER_SECRET     = os.environ['TWITTER_CONSUMER_SECRET']
	ACCESS_TOKEN        = os.environ['TWITTER_ACCESS_TOKEN']
	ACCESS_TOKEN_SECRET = os.environ['TWITTER_ACCESS_TOKEN_SECRET']

	AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	AUTH.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

	api = tweepy.API(AUTH)


	if username is None:
		username = input(Fore.CYAN + 'Enter Username(without @): ' + Fore.RESET)
	if verbose:
		print(Fore.BLACK + 'fetching tweets...')
	#getting user tweets
	try:
		tweets = get_user_tweets(api, username)
		name = api.get_user(username).name
	except Exception:
		print(Fore.RED + '✖ No such user({0}) exists'.format(username))
		sys.exit()


	#processing tweets
	document = cv.transform([' '.join(tweets)])

	if verbose:
		print(Fore.BLACK + 'performing prediction...')

	result = ''
	for model in models[1:]:
		result += model.predict(document)[0]

	#printing MBTI personality type obtained
	print('\n@{0}('.format(username) + Fore.RED + '{0}'.format(name) + Fore.RESET + '): ' + Fore.BLUE + result)

	#deinitializing colorama
	deinit()
# ============= BOOTSTRAPPING ===================
