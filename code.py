#Name: Anooshka Bajaj


import pandas as pd
df=pd.read_csv(r'E:\landslide_data.csv')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.set_option('display.max_columns',50)  


#1
df.drop(columns = ['dates','stationid'],inplace = True)       #to consider only last 7 attributes
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
l = Q1 - 1.5*IQR
h = Q3 + 1.5*IQR
df =  df[~((df<l) | (df>h))]
df.fillna(df.median(),inplace = True)                         #to replace outliers by median

#1(a)
print('\nMinimum Values before normalisation\n',df.min())
print('\nMaximum Values before normalisation\n',df.max())

df_minmax = (df-df.min())/(df.max()-df.min())*(9-3)+3        #doing min-max normalization
print('\nMinimum Values after normalisation\n',df_minmax.min())
print('\nMaximum Values after normalisation\n',df_minmax.max())

#1(b)
print('\nMean before standardization\n',df.mean())
print('\nStandard deviation before standardization\n',df.std())

df_std = (df-df.mean())/df.std()                            #doing standardization
print('\nMean after standardization\n',df_std.mean())
print('\nStandard deviation after standardization\n',df_std.std())

 


#2
mn=np.array([0,0])
cov=np.array([[6.84806467,7.63444163],[7.63444163,13.02074623]])
data=np.random.multivariate_normal(mean=mn, cov=cov, size=1000)


#2(a)
plt.scatter(data[:,0],data[:,1],marker='*')
plt.title('Data Sample')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#2(b)
eigenvalues,eigenvectors=np.linalg.eig(cov)
print('Eigenvalues:\n', eigenvalues)
print('Eigenvectors:\n', eigenvectors)

plt.scatter(data[:,0],data[:,1],marker='*')
X,Y=(eigenvectors)
plt.quiver(0,0,eigenvectors[0][0],eigenvectors[1][0],scale = 3,color = 'red',angles="xy")
plt.quiver(0,0,eigenvectors[0][1],eigenvectors[1][1],scale = 3,color = 'red',angles="xy")
plt.title("Plot of 2D synthetic data and eigen directions")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#2(c)
#(i)
proj=data.dot(eigenvectors)
X=eigenvectors[0][0]*proj[:,0]
Y=eigenvectors[1][0]*proj[:,0]
plt.scatter(data[:,0],data[:,1],marker='*')
plt.quiver(0,0,eigenvectors[0][0],eigenvectors[1][0], scale = 3,color = 'red',angles="xy")
plt.quiver(0,0,eigenvectors[0][1],eigenvectors[1][1],scale = 3,color = 'red',angles="xy")
plt.scatter(X,Y,color = 'm',alpha = 0.2)
plt.title('Projected values onto the first eigen direction')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#(ii)
X = eigenvectors[0][1]*proj[:,1]
Y = eigenvectors[1][1]*proj[:,1]
plt.scatter(data[:,0],data[:,1], marker='*')
plt.quiver(0,0,eigenvectors[0][0],eigenvectors[1][0],scale = 3,color = 'red',angles="xy")
plt.quiver(0,0,eigenvectors[0][1],eigenvectors[1][1],scale = 3,color = 'red',angles="xy")
plt.scatter(X,Y,color = 'm',alpha = 0.2)
plt.title('Projected values onto the second eigen direction')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#2(d)
re_data = np.dot(data,eigenvectors)
rmse = sqrt(mean_squared_error(data, re_data))
print("The value of Reconstruction Error is : ",rmse)




#3

#3(a)
principal_components = PCA(n_components = 2).fit_transform(df_std)
plt.scatter(principal_components[:, 0], principal_components[:, 1], marker='*')
plt.title("Scatter plot of reduced dimensional data")
plt.xlabel("x axis: Dimension 1")
plt.ylabel("y axis: Dimension 2")
plt.show()

variance = principal_components.var(0)
print('\nVariance of the projected data along the two directions:\n',variance)
np.median(principal_components[:, 0])
np.median(principal_components[:, 1])


#3(b)
eigenvalues,eigenvectors = np.linalg.eig(np.cov(df_std,rowvar = False))
edo = eigenvalues.argsort()[::-1]               #to sort eigenvalues in descending order  
eigenvalues = eigenvalues[edo]
eigenvectors = eigenvectors[:,edo]
plt.stem(range(1,8),eigenvalues)
plt.xlabel('Index')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues in descending order')
plt.show()


#3(a) contd.
print('\nEigenvalues of the two directions of projection:\n',eigenvalues[0:2])


#3(c)
RMSE = []
for i in range(1,8):
    pca= PCA(n_components = i)
    d = pca.fit_transform(df_std)
    re = pca.inverse_transform(d)
    rmse = (((((re-df_std)**2).sum(1))**0.5).sum(0))/len(df_std)
    RMSE.append(rmse)
plt.bar(range(1,8),RMSE)
plt.xlabel('l')
plt.ylabel('RMSE')
plt.title('RMSE for different l')
plt.show()
    
    







