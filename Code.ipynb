#Import Relevant Libraries
import pandas as pd 
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import statistics
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn import metrics
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import seaborn as sns 
sns.set() 
from sklearn.cluster import KMeans 
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from numpy import unique

#Read the excel file
data = pd.read_excel('D:/Retail/Online Retail1.xlsx')
print('\n Dimension of Data is : ')
print(data.shape)

#Data Cleaning

#Check for Missing Values
print('\n Missing data : ')
print(data.isnull().sum())

#Removing Records having Blank values in CustomerID
data['CustomerID'].replace('', np.nan, inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)
print('\n Dimension of Data is : ')
print(data.shape)

#Removing Records having Blank values in CustomerID
data['CustomerID'].replace('', np.nan, inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)
print('\n Dimension of Data is : ')
print(data.shape)

#Drop Duplicate Records
data.drop_duplicates()
#print("\n Size of Data Frame after cleaning = {}".format(shape))
print(data.shape)

#Displaying the Cleaned top five records of Dataframe
data['TotalPrice']=data['Quantity']*data['UnitPrice']
print(data.head(5))

df = data.groupby(['InvoiceNo'])['InvoiceNo'].count().reset_index(name='InvoiceCounts')
df1 = data.groupby(['InvoiceNo'])['TotalPrice'].sum().reset_index(name='InvoicePrice')
df=df.join(df1['InvoicePrice'])
print(df.head(5)) 

#View the statistical summary of numerical variables
print('Summary Statistics is :')
df[['InvoiceCounts','InvoicePrice']].describe()

#RFM Analysis

#Calculating Recency
data_recency = data.groupby(by='CustomerID',
                        as_index=False)['InvoiceDate'].max()
data_recency.columns = ['CustomerID', 'LastInvoiceDate']
recent_date = data_recency['LastInvoiceDate'].max()
data_recency['Recency'] = data_recency['LastInvoiceDate'].apply(
    lambda x: (recent_date - x).days)
print(data_recency.head(5))

#Calculating Frequency
frequency_data = data.drop_duplicates().groupby(
    by=['CustomerID'], as_index=False)['InvoiceDate'].count()
frequency_data.columns = ['CustomerID', 'Frequency']
frequency_data.head(5)

#Calculating Monetary Value
data['Total'] = data['UnitPrice']*data['Quantity']
monetary_data = data.groupby(by='CustomerID', as_index=False)['Total'].sum()
monetary_data.columns = ['CustomerID', 'Monetary']
monetary_data.head(5)

#Merging RFM
rf_data = data_recency.merge(frequency_data, on='CustomerID')
rfm_data = rf_data.merge(monetary_data, on='CustomerID').drop(
    columns='LastInvoiceDate')
rfm_data.head(5)

#Ranking Customer’s based upon their RFM score
rfm_data['R_rank'] = rfm_data['Recency'].rank(ascending=False)
rfm_data['F_rank'] = rfm_data['Frequency'].rank(ascending=True)
rfm_data['M_rank'] = rfm_data['Monetary'].rank(ascending=True)
 
# normalizing the rank of the customers
rfm_data['R_rank_norm'] = (rfm_data['R_rank']/rfm_data['R_rank'].max())*100
rfm_data['F_rank_norm'] = (rfm_data['F_rank']/rfm_data['F_rank'].max())*100
rfm_data['M_rank_norm'] = (rfm_data['F_rank']/rfm_data['M_rank'].max())*100
 
rfm_data.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
rfm_data.head(5)

#Calculating RFM score
rfm_data['RFM_Score'] = 0.15*rfm_data['R_rank_norm']+0.28 * \
    rfm_data['F_rank_norm']+0.57*rfm_data['M_rank_norm']
rfm_data['RFM_Score'] *= 0.05
rfm_data = rfm_data.round(2)
rfm_data[['CustomerID', 'RFM_Score']].head(5)

#Rating Customer based upon the RFM score
rfm_data["Customer_segment"] = np.where(rfm_data['RFM_Score'] >
                                      4.5, "Top Customers",
                                      (np.where(
                                        rfm_data['RFM_Score'] > 4,
                                        "High value Customer",
                                        (np.where(
    rfm_data['RFM_Score'] > 3,
                             "Medium Value Customer",
                             np.where(rfm_data['RFM_Score'] > 1.6,
                            'Low Value Customers', 'Lost Customers'))))))
rfm_data['CustomerID'] = rfm_data['CustomerID'].astype(int)
rfm_data[['CustomerID', 'RFM_Score', 'Customer_segment']].head(20)

#Visualizing the customer segments
plt.pie(rfm_data.Customer_segment.value_counts(),
        labels=rfm_data.Customer_segment.value_counts().index,
        autopct='%.0f%%')
plt.show()

3Declare feature vector and target variable
X = df['InvoiceCounts']
y= df['InvoicePrice']

#Split the data Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.80)

#Plot of Train Data
plt.scatter(X_train, y_train)
plt.xlabel('Invoice Counts')
plt.ylabel('Invoice Price')
plt.show()

#Plot of Test Data
plt.scatter(X_test, y_test)
plt.xlabel('Invoice Counts')
plt.ylabel('Invoice Price')
plt.show()

Elbow method to visualize the intertia
data_train = list(zip(X_train, y_train))
data_test = list(zip(X_test, y_test))
inertias = []
​
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_train)
    inertias.append(kmeans.inertia_)
​
plt.plot(range(2,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#K-Means model with three clusters

#lets apply min-max scaling to each row
#define our scaler
scaler = MinMaxScaler()
# scale down our data
data_scaled_train = scaler.fit_transform(data_train)
data_scaled_test = scaler.fit_transform(data_test)
​# see here four rows that are scaled
print( data_scaled_train[0:4])
print(data_scaled_test[0:4])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data_scaled_train)
print('\n kmeans.cluster_centers are :')
print(kmeans.cluster_centers_)
print('\n kmeans.intertia is :')
print(kmeans.inertia_)
plt.scatter(X_train, y_train, c=kmeans.labels_)
plt.show()

#Checking quality of classification by the K-Means model
labels = kmeans.labels_
# check how many of the samples were correctly labeled
correct_labels = sum(labels)
print("\n Result: %d out of %d samples are correctly labeled." % (correct_labels, y.size))
print('\n Accuracy score due to K-Means Model: {0:0.2f}'. format(correct_labels/float(y.size)))

#Evaluating performance of the clustering algorithm
using a Silhouette score
print('\n The Silhouette score for K-Means is : ')
from sklearn.metrics import silhouette_score
print(silhouette_score(data_scaled_train, kmeans.labels_, metric='euclidean'))

#Gaussian Mixture Model

# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(data_train) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
X_principal.head(3)

gmm = GaussianMixture(n_components=3)
gmm.fit(scaled_df)

#Visualizing the clustering of GaussianMixture
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = GaussianMixture(n_components = 3).fit_predict(X_principal), cmap =plt.cm.winter, alpha = 0.6) 
plt.show() 

# Checking quality of classification by the Gaussian Mixture model
labels = gmm.predict(scaled_df)
correct_labels = sum(labels)
print("Result: %d out of %d samples are correctly labeled." % (correct_labels, y.size))
print('\n Accuracy score due to Gaussian Mixture Model: {0:0.2f}'. format(correct_labels/float(y.size)))

#DBSCAN Clustering Model (Distribution Based)

#create a ‘Cluster’ column
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
data_train, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(data_train)

#Computing DB_SCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#Evaluation of DB_SCAN
#Silhouette Coefficient score of more than 0.5 indicating that my model # doesn’t have overlapping clusters or mislabeled data points.
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

#Visualizing the clustering
# Plot result
import matplotlib.pyplot as plt
%matplotlib inline
​
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
​
    class_member_mask = (labels == k)
​
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
​
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
​
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
​
#Hierarchical Clustering
#Dendogram Construction
plt.figure(figsize=(10, 7))
plt.title("DENDOGRAM FOR  DATASET")
dendo_gram = shc.dendrogram(shc.linkage(X, method = "ward"))

#Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster.fit_predict(X)
#Visualising the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')

#Computing Agglomerative Score
print('Agglomerative Score is :')
print(silhouette_score(X,cluster.labels_, metric='euclidean'))


#BIRCH Algorithm
model_br = Birch(threshold=0.01, n_clusters=3)
model_br.fit(X)
yhat_br = model_br.predict(X)
clusters_br = unique(yhat_br)
print("Clusters of Birch",clusters_br)
labels_br = model_br.labels_
plt.scatter(X[:, 0], X[:, 1], c = yhat_br)
score_br = metrics.silhouette_score(X,labels_br)
print("Score of Birch = ", score_br)
