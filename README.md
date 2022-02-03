# Data-Transformation
Attribute normalization, standardization and dimension reduction of data using PCA

I am given the IIT Mandi landslide data-set as a csv file (landslide_data.csv). This data-set contains the readings from various sensors installed at 10 locations around the IIT Mandi campus. These sensors give the details about various factors like temperature, humidity, pressure etc. The given CSV file contains following attributes:\
• dates: date of collection of data.\
• stationid: Indicates the location of the sensor.\
• temperature: Atmospheric temperature around the sensor in Celsius.\
• humidity: The concentration of water vapor present in the air (in g.m-3).\
• pressure: Atmospheric pressure in millibars (mb).\
• rain: Measure of rainfall in ml.\
• lightavgw/o0: The average light throughout the daytime (in lux units).\
• lightmax: The maximum lux count by the sensor.\
• moisture: indicates the water stored in the soil (measured between 0 to 100 percent).

I have written a python program (with pandas) to read the given data and do the following:

1. Replaced the outliers in any attribute with the median of the respective attributes and did the following on outlier corrected data:\
a. Did the Min-Max normalization of the data to scale the attribute values in the range 3 to 9. Found the minimum and maximum values before and after performing the Min-Max normalization of the attributes.\
b. Found the mean and standard deviation of the attributes of the data. Standardized each selected attribute. Compared the mean and standard deviations before and after the standardization.

2. Generated 2-dimensional synthetic data of 1000 samples denoted as data matrix D of size 2x1000. Each sample is independently and identically distributed with 
bi-variate Gaussian distribution with user entered mean values and covariance matrix. Performed the following:\
a. Obtained scatter plot of the data samples.\
b. Computed the eigenvalues and eigenvectors of the covariance matrix and plotted the Eigen directions onto the scatter plot of data.\
c. Projected the data on to the first and second Eigen direction individually and drew both the scatter plots superimposed on Eigen vectors.\
d. Reconstructed the data samples using both eigenvectors and estimated the reconstruction error using mean square error.

3. Performed principle component analysis (PCA) on outlier corrected standardized data and did the following:\
a. Reduced the multidimensional (d = 7) data into lower dimensions (l = 2) and found the variance of the projected data along the two directions and compared with the 
eigenvalues of the two directions of projection. Also obtained the scatter plot of reduced dimensional data.\
b. Plotted all the eigenvalues in the descending order.\
c. Plotted the reconstruction errors in terms of RMSE considering the different values of l (=1, 2, ..., 7). 

