# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

# %% [markdown]
# ### Load the data

# %%
data = pd.read_csv('1.01-Simple-linear-regression.csv')
data.head()

# %% [markdown]
# ### Create the regression
# 
# #### Declare the dependent and independent variables

# %%
x = data['SAT'] # input / feature
y = data['GPA'] # output / target

# %%
x.shape

# %%
y.shape

# %%
# turning 1D arrays into 2D 
x_matrix = x.values.reshape(-1,1)
x_matrix.shape

# %% [markdown]
# #### Regression itself

# %%
reg = LinearRegression()

# %%
reg.fit(x_matrix, y)


