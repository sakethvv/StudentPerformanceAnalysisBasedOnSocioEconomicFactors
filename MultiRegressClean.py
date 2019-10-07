import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd
import numpy as np 
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn import preprocessing


df = pd.read_csv("~/Documents/NEU Course Material/Stat Methods/Project/StudentsPerformance.csv")

df.head()


df=df.rename(columns={
                   'gender':'gender',
                   'race/ethnicity':'group',
                   'parental level of education':'highest_degree',
                   'lunch':'lunch',
                   'test preparation course':'coaching',
                   'math score':'math_score',
                   'reading score':'reading_score',
                   'writing score':'writing_score'
})

df = pd.get_dummies(df, columns=['gender','group','highest_degree','lunch','coaching'],drop_first = False)
df

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = ['math_score','reading_score','writing_score','gender_female','gender_male',
              'group_group A','group_group B','group_group C','group_group D','group_group E',
              'highest_degree_College dropout','highest_degree_associates degree',
              'highest_degree_bachelors degree','highest_degree_high school',
              'highest_degree_high school dropout','highest_degree_masters degree',
              'lunch_free/reduced','lunch_standard','coaching_completed','coaching_none']

df.dtypes

df_cor = df.corr()
corr
df_vif = pd.DataFrame(np.linalg.inv(df.corr().values), index = df_cor.index, columns=df_cor.columns)
df_vif

df['gender_female']=df['gender_female'].astype('category')
df['gender_male']=df['gender_male'].astype('category')
df['group_group A']=df['group_group A'].astype('category')
df['group_group B']=df['group_group B'].astype('category')
df['group_group C']=df['group_group C'].astype('category')
df['group_group D']=df['group_group D'].astype('category')
df['group_group E']=df['group_group E'].astype('category')
df['highest_degree_College dropout']=df['highest_degree_College dropout'].astype('category')
df['highest_degree_associates degree']=df['highest_degree_associates degree'].astype('category')
df['highest_degree_bachelors degree']=df['highest_degree_bachelors degree'].astype('category')
df['highest_degree_high school']=df['highest_degree_high school'].astype('category')
df['highest_degree_high school dropout']=df['highest_degree_high school dropout'].astype('category')
df['highest_degree_masters degree']=df['highest_degree_masters degree'].astype('category')
df['lunch_free/reduced']=df['lunch_free/reduced'].astype('category')
df['lunch_standard']=df['lunch_standard'].astype('category')
df['coaching_completed']=df['coaching_completed'].astype('category')
df['coaching_none']=df['coaching_none'].astype('category')

df.dtypes

X=df.drop('math_score',axis=1)
y=df.math_score
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=2018)

linreg=LinearRegression()
linreg.fit(X_train,y_train)
coeff_df = pd.DataFrame(linreg.coef_, X.columns, columns=['Coefficient'])  
coeff_df
y_pred=linreg.predict(X_test)
r2_score(y_test,y_pred)

new_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
new_df1 = new_df.head(25)
new_df1

new_df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Meam Squared Error is:',mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error is:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('Mean Absolute Error is:',(mean_absolute_error(y_test,y_pred)))
print(linreg.coef_)
print(linreg.intercept_)

y_res = y_pred-y_test
y_res1 = y_res.head(30) 
y_res1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(y_pred,y_res,color='red',edgecolors=(0,0,0))
plt.title("Predicted Math Score vs Actual Math Score")
plt.xlabel("Predicted Score")
plt.ylabel("Residuals")

plt.scatter(y_test,y_res,color='red',edgecolors=(0,0,0))
plt.title("Predicted Math Score vs Actual Math Score")
plt.xlabel("Actual Score")
plt.ylabel("Residuals")

plt.scatter(y_pred,y_test,color='red',edgecolors=(0,0,0))
#plt.plot(y_test,y_pred, color='green')
plt.title("Predicted Math Score vs Actual Math Score")
plt.xlabel("Predicted Score")
plt.ylabel("Actual Score")