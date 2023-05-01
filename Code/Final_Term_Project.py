#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

df = pd.read_csv("Maryland_Property_Data.csv")

np.random.seed(6202)

# Checking amount of features are numeric

numeric_cols = df.select_dtypes(include=['int', 'float'])

num_numeric_cols = len(numeric_cols.columns)
print('Number of numeric columns:', num_numeric_cols)

# Filtering Montgomery County and Residential Types
mcounty_df = df[df.JURSCODE == "MONT"]
mcounty_df.DESCLU.map(lambda x: "Residential" if "Residential" in x else x)
mcounty_df = mcounty_df[mcounty_df.DESCLU == "Residential"]

# Keeping features of interests by looking at data dictionary

mcounty_df = mcounty_df[["ACCTID", #Parcel account number
                         "X", #Longitude
                         "Y", #Latitude
                         "ADDRESS", 
                         "ZIPCODE",
                         "STRUGRAD", #STRUGRAD
                         "YEARBLT", #Year built 
                         "SQFTSTRC", #Square-foot 
                         "TRADATE", #Transfer Date 
                         "CONSIDR1", #Consideration
                         "NFMLNDVL", #New appraised land value
                         "NFMIMPVL"]] #New appraised improved value

# Heatmap of nulls 

ax = plt.axes()
sns.heatmap(mcounty_df.isna().transpose(), cbar=False, ax=ax)
plt.title("Nan heatmap")
plt.xlabel("# of observations")
plt.ylabel("Dataframe columns")
plt.show()

mcounty_df.dropna(inplace=True)

# Reformating Year, Month and Day from transfer date

mcounty_df.TRADATE = mcounty_df.TRADATE.astype(str)
mcounty_df["Year"] = mcounty_df.TRADATE.str.slice(0,4)
mcounty_df["Month"] = mcounty_df.TRADATE.str.slice(4,6)
mcounty_df["Day"] = mcounty_df.TRADATE.str.slice(6,8)
mcounty_df["transfer_date"] = pd.to_datetime(dict(year=mcounty_df.Year, month=mcounty_df.Month, day=mcounty_df.Day))

mcounty_df["Year"] = mcounty_df["Year"].astype(int)
#%%
# EDA 

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))

sns.histplot(mcounty_df.CONSIDR1, ax=axes[0,0], bins=100)
axes[0,0].set_title('Histogram of Consideration')
axes[0,0].set_ylabel("Frequency")
axes[0,0].set_xlabel("U.S. Dollars")

sns.histplot(mcounty_df.NFMLNDVL, ax=axes[0,1], bins=100)
axes[0,1].set_title('Histogram of Land value')
axes[0,1].set_ylabel("Frequency")
axes[0,1].set_xlabel("U.S. Dollars")

sns.histplot(mcounty_df.NFMIMPVL, ax=axes[1,0], bins=100)
axes[1,0].set_title('Histogram of Land Improvements')
axes[1,0].set_ylabel("Frequency")
axes[1,0].set_xlabel("U.S. Dollars")

sns.histplot(mcounty_df.SQFTSTRC, ax=axes[1,1], bins=100)
axes[1,1].set_title('Histogram of Square foot')
axes[1,1].set_ylabel("Frequency")
axes[1,1].set_xlabel("Square foot")

sns.histplot(mcounty_df.YEARBLT, ax=axes[2,0], bins=100)
axes[2,0].set_title('Histogram of Year Built')
axes[2,0].set_ylabel("Frequency")
axes[2,0].set_xlabel("Dates")

sns.histplot(mcounty_df.Year, ax=axes[2,1])
axes[2,1].set_title('Histogram of Transfer Year')
axes[2,1].set_ylabel("Frequency")
axes[2,1].set_xlabel("Dates")

plt.tight_layout()
plt.show()

sns.countplot(data=mcounty_df, x = mcounty_df["STRUGRAD"])
plt.xlabel("Grade")
plt.ylabel("Frequency")
plt.title("Barplot of structure grade")
plt.show()

import geopandas as gpd
from shapely.geometry import Point


mcounty_map = gpd.read_file('/Users/tylerwallett/Downloads/Geographic data_ Zip Codes (Shape File)')

crs = {'init':'epsg:4326'}

geometry = [Point(xy) for xy in zip(mcounty_df['X'], mcounty_df['Y'])]

geo_df = gpd.GeoDataFrame(mcounty_df, 
                          crs = crs, 
                          geometry = geometry)

fig, ax = plt.subplots(figsize=(10, 10))

mcounty_map.plot(ax=ax)

# Group the data by zipcode and calculate the mean value of CONSIDR1 for each zipcode
zipcode_data = geo_df.groupby('ZIPCODE')['CONSIDR1'].mean()

# Join the zipcode data with the mcounty_map GeoDataFrame using the 'zipcode' column
merged_data = mcounty_map.set_index('zipcode').join(zipcode_data)

merged_data.plot(column='CONSIDR1', edgecolor='black', legend=True, ax=ax)

ax.set_axis_off()
plt.title("Average residential consideration by zipcode", loc="center")
plt.show()

mcounty_df.rename(columns={"ACCTID": "Id",
                           "X": "Longitude",
                           "Y": "Latitude",
                           "ADDRESS":"Address",
                           "ZIPCODE": "Zipcode",
                           "STRUGRAD": "Grade",
                           "YEARBLT": "Year_built",
                           "SQFTSTRC": "Sqft",
                           "TRADATE": "Trade_date",
                           "CONSIDR1": "Consideration",
                           "NFMLNDVL": "Land_value",
                           "NFMIMPVL": "Land_improvements"}, inplace= True)

#%%
# Feature Selection

# Correlation Heatmap
plt.figure(figsize=(10,10))
mask = np.triu(np.ones_like(mcounty_df.corr(), dtype=np.bool))
corr = mcounty_df.corr()
sns.heatmap(corr, annot=True, mask = mask, vmin=-1,vmax=1, fmt = '.2f')
plt.title("Correlation heatmap")
plt.show()

# Train test split - Feature Selection

from sklearn.model_selection import train_test_split

x = mcounty_df[["Zipcode",
                "Grade", #Structure grade
                "Year_built", #Year built 
                "Sqft", #Square-foot
                "Land_value", #New appraised land value
                "Land_improvements", #New appraised improved value"
                "Year"]] 

y = mcounty_df["Consideration"]

X_train, X_test, Y_train, Y_test = train_test_split(x ,y, test_size= 0.2, random_state=6202)

# Eigensystem Analysis - conditional number (degree of collinearity)

from numpy.linalg import cond

print(f"Initial conditional number: {cond(X_train).round(2)}", '\n')

X_train2 = X_train.drop(columns='Zipcode')

print(f"Conditional number without regressor `Zipcode`:{cond(X_train2).round(2)}")
print(f"Decrease in conditional number: {(cond(X_train).round(2) - cond(X_train2).round(2)).round(2)}", '\n')

X_train3 = X_train2.drop(columns='Grade')

print(f"Conditional number without regressor `Grade`:{cond(X_train3).round(2)}")
print(f"Decrease in conditional number: {cond(X_train2).round(2) - cond(X_train3).round(2)}", '\n')

X_train4 = X_train3.drop(columns='Year_built')

print(f"Conditional number without regressor `Year_built`:{cond(X_train4).round(2)}")
print(f"Decrease in conditional number: {cond(X_train3).round(2) - cond(X_train4).round(2)}", '\n')

X_train5 = X_train4.drop(columns='Sqft')

print(f"Conditional number without regressor `Sqft`:{cond(X_train5).round(2)}")
print(f"Decrease in conditional number: {cond(X_train4).round(2) - cond(X_train5).round(2)}", '\n')

X_train6 = X_train5.drop(columns='Year')

print(f"Conditional number without regressor `Year`:{cond(X_train6).round(2)}")
print(f"Decrease in conditional number: {cond(X_train5).round(2) - cond(X_train6).round(2)}", '\n')

#Lasso Regression

from sklearn import linear_model

#World 1
model1 = linear_model.LassoLarsCV(cv=10, max_n_alphas=10).fit(X_train,Y_train)

fig, ax = plt.subplots(figsize=(8, 8))

cm1 = iter(plt.get_cmap("tab20")(np.linspace(0, 1, X_train.shape[1])))

for i in range(X_train.shape[1]):
    c = next(cm1)
    ax.plot(
        model1.alphas_,
        model1.coef_path_.T[:, i],
        c=c,
        alpha=0.8,
        label=x.columns[i],
    )

ax.axvline(
    model1.alpha_,
    linestyle="-",
    c="k",
    label="alphaCV",
)

plt.xlim([min(model1.alphas_), max(model1.alphas_)])
plt.ylim([-1, 1.5])
plt.legend()
plt.ylabel("Regression Coefficients")
plt.xlabel("alpha")
plt.title("Lasso regression coefficient paths: Maryland property dataset", wrap =True)
plt.show()
    

# Recursive feature elimination 

# from sklearn import ensemble
# from yellowbrick.features import RFECV
# from matplotlib import image as mpimg

# #### CODES FOR RFE ##########

# fig, ax = plt.subplots(figsize=(6, 4))

# rfe1 = RFECV(ensemble.RandomForestRegressor(n_estimators=100), cv=5)

# rfe1.fit(X_train, Y_train)

# rfe1.rfe_estimator_.ranking_
# rfe1.rfe_estimator_.n_features_
# rfe1.rfe_estimator_.support_
# rfe1.poof()

# importance_scores = rfe1.estimator_.feature_importances_

# # Sort the indices of features based on their importance scores
# sorted_idx = importance_scores.argsort()[::-1]

# # Select the top 4 features based on their sorted indices
# top_features = X_train.columns[sorted_idx][:4]

# fig.savefig('RFE for Maryland property dataset')

#### CODES FOR RFE ##########

# rfew1 = mpimg.imread("RFE for Maryland property dataset.png")
# plt.xticks([])
# plt.yticks([])
# plt.imshow(rfew1)


#%%
# Pre-processing 

from sklearn.preprocessing import QuantileTransformer

x = mcounty_df[["Land_value", #New appraised land value
                "Land_improvements" #New appraised improved value"
                ]] 

y = mcounty_df[["Consideration"]]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.2, random_state=6202)

scaler_x_train = QuantileTransformer(output_distribution="normal", random_state=6202).fit(X_train)

X_train_scaled = scaler_x_train.fit_transform(X_train)

scaler_x_test = QuantileTransformer(output_distribution="normal", random_state=6202).fit(X_test)

X_test_scaled = scaler_x_test.fit_transform(X_test)

scaler_y_train = QuantileTransformer(output_distribution="normal", random_state=6202).fit(Y_train)

Y_train_scaled = scaler_y_train.fit_transform(Y_train)

scaler_y_test = QuantileTransformer(output_distribution="normal", random_state=6202).fit(Y_test)

Y_test_scaled = scaler_y_test.fit_transform(Y_test)

Y_train_scaled = Y_train_scaled.reshape((len(Y_train_scaled),))

Y_test_scaled = Y_test_scaled.reshape((len(Y_test_scaled),))

#%%
# Subplots of raw values

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,8))

sns.kdeplot(X_train.iloc[:,0], ax = axes[0])
axes[0].set_title('Density plot of Land Value')
axes[0].set_ylabel("Density")
axes[0].set_xlabel("Normalized values")

sns.kdeplot(X_train.iloc[:,1], ax = axes[1])
axes[1].set_title('Density plot of Land Improvements')
axes[1].set_ylabel("Density")
axes[1].set_xlabel("Normalized values")

sns.kdeplot(Y_train, ax = axes[2], legend=False)
axes[2].set_title('Density plot of Consideration')
axes[2].set_ylabel("Density")
axes[2].set_xlabel("Normalized values")

plt.tight_layout()
plt.show()

#%%
# Subplots of pre-processed values

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,8))

sns.kdeplot(X_train_scaled[:,0], ax = axes[0])
axes[0].set_title('Density plot of Land Value')
axes[0].set_ylabel("Density")
axes[0].set_xlabel("Normalized values")

sns.kdeplot(X_train_scaled[:,1], ax = axes[1])
axes[1].set_title('Density plot of Land Improvements')
axes[1].set_ylabel("Density")
axes[1].set_xlabel("Normalized values")

sns.kdeplot(Y_train_scaled, ax = axes[2])
axes[2].set_title('Density plot of Consideration')
axes[2].set_ylabel("Density")
axes[2].set_xlabel("Normalized values")

plt.tight_layout()
plt.show()

#%%
# Models

def logsig(n):
    result = np.empty_like(n)
    for i in range(n.shape[0]):
        result[i,0] = 1 / (1 + np.exp(-n[i,0]))
    return result

#%%
def backpropagation_ftp(inputs, targets, s_size, alpha, n_iter):
    np.random.seed(6202)
    inputs = np.concatenate([inputs] * n_iter)
    targets = np.concatenate([targets] * n_iter)

    w_init_1 = np.matrix(np.random.rand(s_size,inputs.shape[1]))
    b_init_1 = np.random.rand(s_size, 1)

    w_init_2 = np.matrix(np.random.rand(1,w_init_1.shape[0]))
    b_init_2 = np.random.rand(1, 1)
    
    error_l = []
    w_1history = []
    w_2history = []
    b_1history = []
    b_2history = []
    output = []

    for i in range(len(inputs)):
        
        # FEEDFOWARD PART
        
        a_init = inputs[i:i+1].reshape((inputs.shape[1],1))
        
        n1 = (w_init_1 @ a_init + b_init_1)
        
        a1 = logsig(n1)
        
        a2 = ((w_init_2 @ a1) + b_init_2).item()
        
        target = targets[i:i+1].reshape(1,1).item()

        error = target - a2
  
        # BACKPROPAGATION FOR LOOP

        f_1_1 = (np.ones((s_size,1)) - a1)
        f_1 = (np.matrix(np.diag([f_1_1[i,0] for i in range(s_size)]))) @ (np.matrix(np.diag([a1[i,0] for i in range(s_size)])))
        f_2 = 1

        s2 = -2 * f_2 * error
        s1 = f_1 @ (w_init_2.T * s2)

        # WEIGHT UPDATE 

        w_init_2 = w_init_2 - (alpha * s2 * a1.T)
        b_init_2 = b_init_2 - (alpha * s2)

        w_init_1 = w_init_1 - (alpha * s1 * a_init.T)
        b_init_1 = b_init_1 - (alpha * s1)
        
        error_l.append(error)
        w_1history.append(w_init_1)
        w_2history.append(w_init_2)
        b_1history.append(b_init_1)
        b_2history.append(b_init_2)
        output.append(a2)
    return error_l, w_1history, w_2history, b_1history, b_2history, output

error, w_1, w_2, b_1, b_2, output = backpropagation_ftp(inputs = X_train_scaled, targets = Y_train_scaled, s_size = 10, alpha =1e-02, n_iter=10)

#%%
import math

error = np.array(error)
error_sq = (error ** 2).sum()
denominator = np.cumsum((np.concatenate([Y_train_scaled] * 10)  - np.concatenate([Y_train_scaled] * 10) .mean()) **2)[-1]
r_2 = 1 - (error_sq/denominator)
print(r_2)

error_sq_plot = error ** 2
print(error_sq_plot.sum())

lower = error_sq_plot.mean() - (1.95 * (error_sq_plot.std()/math.sqrt(len(error_sq_plot))))
upper = error_sq_plot.mean() + (1.95 * (error_sq_plot.std()/math.sqrt(len(error_sq_plot))))

print(f"Mean error_sq: {error_sq_plot.mean()}")
print(f"Upper confidence: {upper}")
print(f"Lower confidence: {lower}")

plt.loglog(error_sq_plot)
plt.show()
plt.plot(error_sq_plot)
plt.show()

#%%
sns.kdeplot(Y_train_scaled, legend=True)
sns.kdeplot(output, legend=True)
plt.show()

#%%
def predict(inputs, targets, w_1,b_1,w_2,b_2):
    error_l = []
    w_1history = []
    w_2history = []
    b_1history = []
    b_2history = []
    output = []
    
    for i in range(len(inputs)):
        
        # FEEDFOWARD PART
        
        a_init = inputs[i:i+1].reshape((inputs.shape[1],1))
        
        n1 = (w_1 @ a_init + b_1)
        
        a1 = logsig(n1)
        
        a2 = ((w_2 @ a1) + b_2).item()
        
        target = targets[i:i+1].reshape(1,1).item()

        error = target - a2
        
        error_l.append(error)
        w_1history.append(w_1)
        w_2history.append(w_2)
        b_1history.append(b_1)
        b_2history.append(b_2)
        output.append(a2)
    return error_l, w_1history, w_2history, b_1history, b_2history, output
    
error, w_1, w_2, b_1, b_2, output = predict(X_test_scaled, Y_test_scaled, w_1[-1], b_1[-1], w_2[-1], b_2[-1])

#%%
error = np.array(error)
error_sq = (error ** 2).sum()
denominator = np.cumsum((Y_test_scaled - Y_test_scaled.mean()) **2)[-1]
r_2 = 1 - (error_sq/denominator)
print(r_2)

error_sq_plot = error ** 2
print(error_sq_plot.sum())

lower = error_sq_plot.mean() - (1.95 * (error_sq_plot.std()/math.sqrt(len(error_sq_plot))))
upper = error_sq_plot.mean() + (1.95 * (error_sq_plot.std()/math.sqrt(len(error_sq_plot))))

print(f"Mean error_sq: {error_sq_plot.mean()}")
print(f"Upper confidence: {upper}")
print(f"Lower confidence: {lower}")

plt.plot(error_sq_plot)
plt.show()
plt.loglog(error_sq_plot)
plt.show()

#%%
sns.kdeplot(Y_test_scaled, legend=True)
sns.kdeplot(output, legend=True)
plt.show()
#%%

output_post = scaler_y_test.inverse_transform(np.array(output).reshape(-1, 1))

#%%
from matplotlib.animation import FuncAnimation 

#setup figure
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)

#rolling window size
repeat_length = 25

ax.set_xlim([0,repeat_length])
ax.set_ylim([0,2.5e6])

#set figure to be modified
im, = ax.plot([], [])
im2, = ax.plot([], []) 

def func(n):
    im.set_xdata(np.arange(n))
    im.set_ydata(Y_test[0:n])
    im2.set_xdata(np.arange(n))  
    im2.set_ydata(output_post[0:n]) 
    if n>repeat_length:
        lim = ax.set_xlim(n-repeat_length, n)
    else:
        lim = ax.set_xlim(0,repeat_length)
    return im

ani = FuncAnimation(fig, func, frames=500, interval=30, blit=False)

plt.legend(['Y_test', 'a2'], loc = 'lower left')
plt.ylabel("Magnitude")
plt.xlabel("Observations")
plt.title("Homemade MLPRegressor S1=10 learning")
plt.show()

ani.save('MLPRegressorModel_Homemade.gif',writer='ffmpeg', fps=10)

#%%
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor(max_depth=10, random_state=6202).fit(X_train_scaled, Y_train_scaled)

print(model2.score(X_test_scaled, Y_test_scaled).round(4))

#%%
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


X_train_scaled = sm.add_constant(X_train_scaled)
model3 = OLS(Y_train_scaled, X_train_scaled).fit()

print(model3.summary())

X_train_scaled = X_train_scaled[:,1:]


#%%
from sklearn.neural_network import MLPRegressor

model4 = MLPRegressor(hidden_layer_sizes=(10),
                      activation='logistic',
                      random_state=6202).fit(X_train_scaled, Y_train_scaled)

print(model4.score(X_test_scaled, Y_test_scaled))

predictions_scaled = model4.predict(X_test_scaled)

predictions = scaler_y_test.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
#%%

from matplotlib.animation import FuncAnimation 

#setup figure
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)

#rolling window size
repeat_length = 25

ax.set_xlim([0,repeat_length])
ax.set_ylim([0,2.5e6])

#set figure to be modified
im, = ax.plot([], [])
im2, = ax.plot([], []) 

def func(n):
    im.set_xdata(np.arange(n))
    im.set_ydata(Y_test[0:n])
    im2.set_xdata(np.arange(n))  
    im2.set_ydata(predictions[0:n]) 
    if n>repeat_length:
        lim = ax.set_xlim(n-repeat_length, n)
    else:
        lim = ax.set_xlim(0,repeat_length)
    return im

ani = FuncAnimation(fig, func, frames=500, interval=30, blit=False)

plt.legend(['Y_test', 'Predictions'], loc = 'lower left')
plt.ylabel("Magnitude")
plt.xlabel("Observations")
plt.title("Scikit-Learn MLPRegressor S1=10 learning")
plt.show()

ani.save('MLPRegressorModel_Post.gif',writer='ffmpeg', fps=10)

#%%
from matplotlib.ticker import MaxNLocator

error_sq_scaled = np.array(Y_test_scaled - predictions_scaled) ** 2

lower = error_sq_scaled.mean() - (1.95 * (error_sq_scaled.std()/math.sqrt(len(error_sq_scaled))))
upper = error_sq_scaled.mean() + (1.95 * (error_sq_scaled.std()/math.sqrt(len(error_sq_scaled))))

print(f"Mean error_sq: {error_sq_scaled.mean().round(2)}")
print(f"Upper confidence: {upper.round(2)}")
print(f"Lower confidence: {lower.round(2)}")

error = Y_test - predictions
error = np.array(error)
lower = error.mean() - (1.95 * (error.std()/math.sqrt(len(error))))
upper = error.mean() + (1.95 * (error.std()/math.sqrt(len(error))))

print(f"Mean error: {error.mean().round(2)}$")
print(f"Upper confidence: {upper.round(2)}$")
print(f"Lower confidence: {lower.round(2)}$")

plt.plot(model4.loss_curve_)
plt.xlabel("# of iterations")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel("Loss")
plt.title("Scikit-Learn MLPRegressor S1=10 Loss curve")
plt.show()
#%%

# Sample Approximation: 

Case = mcounty_df.sample(1, random_state= 6202)

Case_x = X_test[X_test.index == 739805]

Case_y = Y_test[X_test.index == 739805]

print(f"Land Value: {Case_x.iloc[0,0]}")
print(f"Land Improvements: {Case_x.iloc[0,1]}")
print(f"Consideration: {Case_y.iloc[0,0]}")
print(f"Prediction: {output_post[0].item()}")

# %%
