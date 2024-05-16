'''Jenny Wu
PODS Capstone
'''
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.linear_model import LogisticRegression
from scipy.special import expit 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, mean_squared_error, accuracy_score, classification_report

#first create a RNG
mySeed = 21
np.random.seed(mySeed)
random.seed(mySeed)
#define alpha for entire file
alpha = 0.05
#Load csv
dataset = pd.read_csv("spotify52kData.csv")
#-------------------------------------------------------------------------------------------------------------------

#Question 1
#sub dataframe with the 10 song features
song_features_10 = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']
features_df = dataset[song_features_10]
#turn them into zscores
z_score_features = stats.zscore(features_df)
#plotting each feature as histogram
fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize=(15, 6))
fig.suptitle('Distribution of Song Features (Z Scored)')
for i, feature in enumerate(z_score_features.columns):
    row = i//5
    col = i%5
    numbins = len(np.unique(z_score_features[feature]))
    axes[row, col].hist(z_score_features[feature], bins = 100)
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
# Adjust layout
plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------------------------------------------
#Question 2
#dataframe of duration and popularity
len_popularity = (dataset[['duration', 'popularity']]).dropna()
#plot the points
plt.scatter(len_popularity['duration'], len_popularity['popularity'], s = 1)
plt.title("Duration and popularity")
plt.xlabel('Duration (milliseconds)')
plt.ylabel('Popularity (0 to 100)')
plt.ticklabel_format(style='plain', axis='x')
plt.show()
#correlation between variables
len_popularity_corr = len_popularity.corr()
#nonlinear, monotonic
spearman_value, p_value = spearmanr(len_popularity['duration'], len_popularity['popularity'])
#----------------------------------------------------------------------------------------------------------------
#Question 3
#defining the data needed for this question
explicit = dataset[dataset['explicit'] == True]['popularity']
non_explicit = dataset[dataset['explicit'] == False]['popularity']
#descriptive stats for each group
explicit_summ_stats = explicit.describe()
non_explicit_summ_stats = non_explicit.describe()
#visualizing explicit distribution
plt.hist(explicit, edgecolor = 'white')
plt.axvline(explicit_summ_stats[1], color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(explicit_summ_stats[5], color='green', linestyle='dashed', linewidth=2, label='Median')
plt.title('Explicit Song Popularity')
plt.xlabel('Popularity (0 to 100)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#visualize non explicit distribution 
plt.hist(non_explicit, edgecolor = 'white')
plt.axvline(non_explicit_summ_stats[1], color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(non_explicit_summ_stats[5], color='green', linestyle='dashed', linewidth=2, label='Median')
plt.title('Non-Explicit Song Popularity')
plt.xlabel('Popularity (0 to 100)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#conducting mann whitney u test first
u_value, p_value = stats.mannwhitneyu(explicit, non_explicit)
#-------------------------------------------------------------------------------------------------------------
#Question 4
#splitting the data into major and minor keys
major_group = dataset[dataset['mode'] == 1]['popularity'].dropna()
minor_group = dataset[dataset['mode'] == 0]['popularity'].dropna()
#summary stats of both groups
major_summary_stats = major_group.describe()
minor_summary_stats = minor_group.describe()
#visualizing major key distribution 
plt.hist(major_group, edgecolor = 'white')
plt.axvline(major_summary_stats[1], color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(major_summary_stats[5], color='green', linestyle='dashed', linewidth=2, label='Median')
plt.title('Major Key Popularity')
plt.xlabel('Popularity (0 to 100)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#visualizing minor key distribution 
plt.hist(minor_group,edgecolor = 'white')
plt.axvline(minor_summary_stats[1], color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(minor_summary_stats[5], color='green', linestyle='dashed', linewidth=2, label='Median')
plt.title('Minor Key Popularity')
plt.xlabel('Popularity (0 to 100)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#mann whitney u because it's not normally distributed
u_value,p_value = stats.mannwhitneyu(major_group, minor_group)
#---------------------------------------------------------------------------------------------------------------
#Question 5
#visualize variables first
plt.scatter(dataset['energy'], dataset['loudness'],  s = 1)
plt.title('Relationship Between Energy and Loudness')
plt.xlabel('Energy (0 to 1)')
plt.ylabel('Loudness (db)')
#look at variables' correlations
dataset_corr_matrix = np.corrcoef(dataset[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']], rowvar = False)
#dataframe for this question after looking at the correlations, and standardizing since it's more than 1 IV
energy_loud = dataset[['energy', 'acousticness', 'instrumentalness', 'loudness']].dropna()
energy_loud_zscore = stats.zscore(energy_loud)
X = energy_loud_zscore[['energy', 'acousticness', 'instrumentalness']]
y = energy_loud_zscore['loudness']
#split into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=mySeed)
#using lasso regression
alph = 0.1
numIt = 10000 
lasso = Lasso(max_iter = numIt)
lasso.set_params(alpha=alph)
lasso.fit(scale(X_train), y_train)
lasso_betas = lasso.coef_
print(mean_squared_error(y_test, lasso.predict(X_test)))
print(r2_score(y_test, lasso.predict(X_test)))
#--------------------------------------------------------------------------------------------------------------
#Question 6
#dataset for this question, normalizing data, set x and y variables
X_features = dataset[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
             'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']]
X_features_z = stats.zscore(X_features)
X = X_features_z.iloc[:, 0:10]
y = X_features_z.iloc[:,10]
#individal random forest regressor because it's not a strong linear relationship
for each in X.columns:
    x = X[each].to_numpy().reshape(-1,1)
    #split test and train set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= mySeed)
    #eandom forest regressor model
    feature_rf_model = RandomForestRegressor(n_estimators = 100, random_state = mySeed).fit(x_train, y_train)
    #popularity prediction, r2 and rmse
    y_pred = feature_rf_model.predict(x_test)
    RMSE_score = np.sqrt(mean_squared_error(y_test, y_pred))
    rsq = r2_score(y_test, y_pred)   
    #plot prediction vs actual
    plt.scatter(y_pred, y_test, s = 1)
    plt.xlabel('Prediction from model')
    plt.ylabel('Actual popularity')
    plt.title('{} R^2 = {:.3f}, RMSE = {:.3f}'.format(each, rsq, RMSE_score))
    plt.show()
#------------------------------------------------------------------------------------------------------------------
#question 7
#dataframe for this question
feat_pop = dataset[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
             'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']].dropna()
feat_pop_z = stats.zscore(feat_pop)
X = feat_pop_z.iloc[:, 0:10]
y = feat_pop_z.iloc[:, 10]
#split dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = mySeed)
#random forest model with all features
random_forest_model = RandomForestRegressor(n_estimators = 100, random_state = mySeed).fit(X_train, y_train)
#prediction, r2, rmse
y_pred = random_forest_model.predict(X_test)
r2_full= r2_score(y_test, y_pred)
RMSE_full = mean_squared_error(y_test, y_pred)
#graphing prediction vs actual 
plt.scatter(y_pred, y_test, s = 1)
plt.xlabel('Prediction from model')
plt.ylabel("Actual popularity")
plt.title('R^2 = {:.3f}, RMSE = {:.3f}'.format(r2_full, RMSE_full))
plt.show()
#------------------------------------------------------------------------------------------------------------------
#Question 8
#dataset with the features, drop nan if any
features = dataset[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo']].dropna()
#standardizing data 
zscores_features = stats.zscore(features)
#pca with zscored data
pca = PCA().fit(zscores_features) 
#eigenvalues
eigVals = pca.explained_variance_
#loadings for each factor
loadings = pca.components_
#new coordinate system
rotatedData = pca.fit_transform(zscores_features)
#variance explained for pca factors
varExplained = eigVals/sum(eigVals)*100
#visualizing pca factors, eigenvalues
numFeatures = 10 
x = np.linspace(1,numFeatures,numFeatures)
plt.bar(x, eigVals)
plt.plot([0,numFeatures],[1,1],color='orange')
plt.title('Eigenvalues for Song Features')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
#visualizing loadings
whichPrincipalComponent = 2
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
plt.title('Loadings for Component 3')
plt.xlabel('Feature')
plt.ylabel('Loading')
plt.show()
#clustering process
x = np.column_stack((rotatedData[:,0],rotatedData[:,1], rotatedData[:,2], rotatedData[:,3]))
#looping over 9 clusters
numClusters = 9 
sSum = np.empty([numClusters,1])*np.NaN 
# Compute kMeans for each k (code from coding session)
for ii in range(2, numClusters+2): 
    # compute kmeans using scikit
    kMeans = KMeans(n_clusters = int(ii), random_state = mySeed).fit(x) 
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_ 
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(x,cId) 
    # take the sum
    sSum[ii-2] = sum(s)  
# Plot the sum of the silhouette scores as a function of the number of clusters
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()
#-----------------------------------------------------------------------------------------------------------------
#Question 9
#part 1: using only valence
key_valence_df = dataset[['valence', 'mode']].dropna()
x_variable = key_valence_df['valence'].to_numpy().reshape(len(key_valence_df),1) 
y_variable = key_valence_df['mode'].to_numpy()
#split data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x_variable, y_variable, test_size=0.2, random_state= mySeed)
#form logistic model
logistic_model = LogisticRegression().fit(x_train, y_train)
#finding y_prediction
x1 = np.linspace(min(x_test), max(x_test), len(y_test))
y_predict = x1*logistic_model.coef_ + logistic_model.intercept_
sigmoid = expit(y_predict)
#plot logistic regression
plt.plot(x1, sigmoid.ravel(), color='red', label='Logistic Regression')
plt.scatter(x_test, y_test, label = 'Values')
plt.hlines(0.5, min(x_test), max(x_test), color = 'gray', linestyles= 'dotted')
plt.xlabel('Valence')
plt.ylabel('Probability of Major/Minor Key')
plt.yticks(np.array([0,1]))
plt.title('Logistic Regression for Major/Minor Key Prediction')
plt.legend()
plt.show()
#roc score
roc_score = roc_auc_score(y_test, sigmoid)
# Plot the ROC curve
fp_rate, tp_rate, thresholds = roc_curve(y_test, y_predict)
plt.plot(fp_rate, tp_rate, color='red', label='AUC = {:.2f}'.format(roc_score))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
#part 2: using pca to find which features are most important
most_rel_features = dataset[['danceability', 'speechiness', 'liveness', 'valence', 'mode']].dropna()
X_multi = most_rel_features[['danceability', 'speechiness', 'liveness', 'valence']].to_numpy()
X_multi_zscale = stats.zscore(X_multi)
y_multi = most_rel_features['mode'].to_numpy()
#test and train set
X_train, X_test, y_train, y_test = train_test_split(X_multi_zscale, y_multi, test_size=0.2, random_state= mySeed)
#model with four features
logistic_model_multi = LogisticRegression().fit(X_train, y_train)
#y predictions (another way to plot roc curve with the aid of chatgpt)
y_probabilities = logistic_model_multi.predict_proba(X_test)[:, 1]
#roc score
roc_score_multi = roc_auc_score(y_test, y_probabilities)
# Plot the ROC curve
fp_rate, tp_rate, thresholds = roc_curve(y_test, y_probabilities)
plt.plot(fp_rate, tp_rate, color='red', label='AUC = {:.2f}'.format(roc_score_multi))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
#---------------------------------------------------------------------------------------------------------------------
#Question 10
#mapping genre with a numerical value
unique_genres = dataset['track_genre'].dropna().unique()
genre_num_value = {string: i + 1 for i, string in enumerate(unique_genres)}
#y variable
y_labels = dataset['track_genre'].map(genre_num_value)
#using the standardized feature data variable from Question 8
X_features = zscores_features
#split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=mySeed)
#random forest classifier with features directly
genre_classifier = RandomForestClassifier(random_state=mySeed).fit(X_train, y_train)
#y predictions
y_pred_classify = genre_classifier.predict(X_test) 
#accuracy score for classifier
model_accuracy = accuracy_score(y_test, y_pred_classify)
#resi;ts
print("Classifier with features directly accuracy:", model_accuracy)
print(classification_report(y_test, y_pred_classify))




