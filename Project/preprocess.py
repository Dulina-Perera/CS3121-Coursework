# %%
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import SimpleImputer, KNNImputer
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# %%
Marvellous = pd.read_csv("employees.csv")

# %%
print(Marvellous.info())

# %%
display(Marvellous.head())

# %%
display(Marvellous.describe())

# %%
Marvellous.isnull().sum()

# %% [markdown]
# Check for columns with all missing values
# 

# %%
# Check for columns with all missing values and remove them
Marvellous.dropna(axis=1, how='all',inplace=True)

# %% [markdown]
# We have both Religion_ID and Religion columns. Both are same. We can remove 'Religion' column and keep  'Religion_ID' value by replacing 5 with 2.
# 
# 1 – Buddhist
# 2 – Muslim
# 3 – Hindu
# 4 - catholic

# %%
# Remove the 'Religion' column
Marvellous = Marvellous.drop(columns=['Religion'])
# Replace the value '5' with '2' in the 'Religion_ID' column
Marvellous['Religion_ID'] = Marvellous['Religion_ID'].replace(5, 2)

# %% [markdown]
# we have both Designation_ID  and Designation columns. Both are same. We can remove Designation_ID  column

# %%
# Remove the 'Designation_ID' column
Marvellous = Marvellous.drop(columns=['Designation_ID'])

# %%
def get_unique_values_info(data):
    """
    Function to create a DataFrame storing unique values count and unique values for each feature.

    Parameters:
    - data: pandas DataFrame

    Returns:
    - unique_values_info: pandas DataFrame with columns ['Feature', 'Unique Values Count', 'Unique Values']
    """
    unique_values_info = pd.DataFrame(columns=['Feature', 'Unique Values Count', 'Unique Values'])

    for column in data.columns:
        unique_values = data[column].dropna().unique()
        unique_values_count = len(unique_values)
        unique_values_info.loc[len(unique_values_info)] = [column, unique_values_count, unique_values]

    return unique_values_info

# %%
unique_values_info = get_unique_values_info(Marvellous)
display(unique_values_info)

# %%
# unique features in the  Reporting_emp_1	 feature
print(Marvellous['Date_Joined'].unique())
print(Marvellous['Date_Resigned'].unique())
print(Marvellous['Inactive_Date'].unique())
print(Marvellous['Reporting_emp_1'].unique())
print(Marvellous['Designation'].unique())
print(Marvellous['Year_of_Birth'].unique())

# %% [markdown]
# There are missing values present in the type of '\\N'.So we have converted them to Nan

# %%
# Convert each of these values to NaN.
Marvellous = Marvellous.where(Marvellous != '\\N')

print(Marvellous.isnull().sum())

# %% [markdown]
# Analysing Missing Values

# %%

def analyze_missing_values(data):
    # Calculate the number of missing values in each column
    missing_values_count = data.isnull().sum()

    # Calculate the percentage of missing values in each column
    missing_values_percentage = (missing_values_count / len(data)) * 100

    # Get the data types of each column
    data_types = data.dtypes

    # Create a DataFrame to store the results
    missing_values_info = pd.DataFrame({
        'Column': missing_values_count.index,
        'Data Type': data_types,
        'Missing Values Count': missing_values_count,
        'Missing Values Percentage': missing_values_percentage
    })

    # Display the missing_values_info DataFrame
    print("Missing Values Information:")
    display(missing_values_info)

    # Create a bar plot for missing values percentage
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values_percentage, y=missing_values_percentage.index)
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Columns')
    plt.title('Percentage of Missing Values in Each Column')
    plt.show()


# %%
# analyze_missing_values(Marvellous)

# %% [markdown]
# Dropping features with missing value percentages exceeding 90%

# %%
#remove the column with all missing values
Marvellous.dropna(axis=1, how='all',inplace=True)

# %%
#drop the feature reporting_emp_1
Marvellous.drop('Reporting_emp_1', axis=1, inplace=True)

# %%
Marvellous.shape

# %% [markdown]
# Converting values such as '0000','0000-00-00','0000-00-00' in Year_of_Birth,Date_Resigned and Inactive_Date to Nan

# %%
Marvellous.Year_of_Birth = Marvellous.Year_of_Birth.where(Marvellous.Year_of_Birth != "'0000'")
Marvellous.Date_Resigned = Marvellous.Date_Resigned.where(Marvellous.Date_Resigned != '0000-00-00')
Marvellous.Inactive_Date = Marvellous.Inactive_Date.where(Marvellous.Inactive_Date != '0000-00-00')

print(Marvellous.isnull().sum())

# %%
Marvellous.head()

# %%
Marvellous.info()

# %% [markdown]
# Seperation of Categorical and Numerical Values

# %%
numeric_data = Marvellous.select_dtypes(include=['float64', 'int64'])
categorical_data = Marvellous.select_dtypes(include=['object'])

# %% [markdown]
# Handling outliers

# %%
def plot_outliers(data):
    """
    Function to plot outliers in numerical features using box plots.

    Parameters:
    - data: pandas DataFrame containing numerical features.
    """
    num_cols = 5
    num_features = len(data.columns)
    num_rows = (num_features - 1) // num_cols + 1
    
    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for i, column in enumerate(data.columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(y=data[column], color='skyblue', orient='v')
        plt.title(column)

    plt.tight_layout()
    plt.show()


# %%
#seaborne library box plot to visualize outliers
# plot_outliers(numeric_data)

# %%
# plot_outliers(numeric_data)

# %%
numeric_data.head(50)

# %% [markdown]
# Checking Skewness

# %%
#create histrograms for all the numerical values


def plot_histograms(data):
    """
    Function to plot histograms with bars for numerical features.

    Parameters:
    - data: pandas DataFrame containing numerical features.
    """
    num_cols = 5
    num_features = len(data.columns)
    num_rows = (num_features - 1) // num_cols + 1
    
    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for i, column in enumerate(data.columns):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.hist(data[column], color='skyblue', bins=20)
        plt.title(column)

    plt.tight_layout()
    plt.show()


# %%
# plot_histograms(numeric_data)

# %%
def check_skewness(data, threshold=0.5):
    """
    Function to check the skewness of each numeric feature in the DataFrame.

    Parameters:
    - data: pandas DataFrame containing numeric features.
    - threshold: Threshold value for skewness (default is 0.3).

    Returns:
    - skewed_features: List of features that are considered skewed.
    """
    skewed_features = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = data[column].skew()
        if abs(skewness) > threshold:
            skewed_features.append(column)
    
    return skewed_features

# %%
skewed_features = check_skewness(numeric_data)
print("Skewed features:", skewed_features)

# %% [markdown]
# Imputaion for missing values in the numerica data

# %%
# Handle missing values for numerical skewed features
skewed_columns = ['Employee_Code','Religion_ID']
numerical_imputer_skewed = SimpleImputer(strategy='median')
numeric_data[skewed_columns] = numerical_imputer_skewed.fit_transform(numeric_data[skewed_columns])

# %%
# Handle missing values for numerical non_skewed features
non_skewed_columns = ['Employee_No']
numerical_imputer_non_skewed = SimpleImputer(strategy='mean')

numeric_data[non_skewed_columns] = numerical_imputer_non_skewed.fit_transform(numeric_data[non_skewed_columns])

#convert 'Employee_No  and 'Employee_Code'  to int64
numeric_data['Employee_No'] = numeric_data['Employee_No'].astype('int64') 
numeric_data['Employee_Code'] = numeric_data['Employee_Code'].astype('int64')
numeric_data['Religion_ID'] = numeric_data['Religion_ID'].astype('int64')

# %%
numeric_data = pd.DataFrame(numeric_data, columns=numeric_data.columns)

# %%
numeric_data.isnull().sum()

# %%
numeric_data.head(20)


# %% [markdown]
# Categorical Data 

# %%
categorical_data.head()

# %%
categorical_data.isnull().sum()

# %%
def get_most_recent_date(df, column_name):
    """
    Function to compute the most recent date from a column in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column_name: Name of the column containing date values

    Returns:
    - most_recent_date: Most recent date in the column (in the format "12/8/1993")
    """
    # Convert the column to datetime format
    df[column_name] = pd.to_datetime(df[column_name])

    # Find the most recent date and format it
    most_recent_date = df[column_name].max().strftime("%m/%d/%Y")
    
    return most_recent_date


# %%
most_recent_inactive_date = get_most_recent_date(categorical_data, 'Inactive_Date')
print(most_recent_inactive_date)

# %%
# Convert 'Inactive_Date' column to datetime format
categorical_data['Inactive_Date'] = pd.to_datetime(categorical_data['Inactive_Date'])

# Fill missing values with '2022-02-07'
categorical_data['Inactive_Date'].fillna(most_recent_inactive_date, inplace=True)


# %%
most_recent_resigned_date = get_most_recent_date(categorical_data, 'Date_Resigned')
print(most_recent_resigned_date)

# %%
# Convert 'Inactive_Date' column to datetime format
categorical_data['Date_Resigned'] = pd.to_datetime(categorical_data['Date_Resigned'])

# Fill missing values with '2022-02-07'
categorical_data['Date_Resigned'].fillna(most_recent_resigned_date, inplace=True)

# %% [markdown]
# Imputing inference-based approach for missing values in the “Year_of_Birth” and “Marital_Status”

# %%
def impute_knn(df, features):
    ''' Inputs: pandas DataFrame containing feature matrix '''
    ''' Outputs: DataFrame with NaN imputed '''

    # Separate dataframe into numerical/categorical
    numeric_df = df.select_dtypes(include=[np.number])          
    categorical_df = df.select_dtypes(exclude=[np.number])      

    # Impute missing values in specified numerical features using K-nearest neighbors regression
    for col in features:
        if col in numeric_df.columns and numeric_df[col].isna().any():
            imp_test = numeric_df[numeric_df[col].isna()]   
            imp_train = numeric_df.dropna()                
            model = KNeighborsRegressor(n_neighbors=5)      
            knr = model.fit(imp_train.drop(columns=[col]), imp_train[col])
            numeric_df.loc[df[col].isna(), col] = knr.predict(imp_test.drop(columns=[col]))

    # Impute missing values in specified categorical features with mode
    for col in features:
        if col in categorical_df.columns and categorical_df[col].isna().any():
            mode_value = categorical_df[col].mode()[0]
            categorical_df[col].fillna(mode_value, inplace=True)

    return pd.concat([numeric_df, categorical_df], axis=1)

# %%
features = ['Year_of_Birth','Marital_Status' ]
categorical_data = impute_knn(categorical_data,features)

# %%
# Handle missing values for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_data[categorical_data.columns] = categorical_imputer.fit_transform(categorical_data)

# %%
categorical_data.isnull().sum()

# %%
categorical_data.head()

# %%
unique_values_info = get_unique_values_info(categorical_data)
display(unique_values_info)

# %% [markdown]
# Remove unwanted Features

# %%
categorical_data = categorical_data.drop(columns=['Name'])
categorical_data = categorical_data.drop(columns=['Title'])

# %% [markdown]
# Creating new categorical features

# %% [markdown]
# Active_Duration(months) = Inactive_Date  - Date_Joined

# %%
# Convert 'Date_Joined' and 'Inactive_Date' columns to datetime format
# Convert 'Date_Joined' and 'Inactive_Date' columns to datetime format
categorical_data['Date_Joined'] = pd.to_datetime(categorical_data['Date_Joined'])
categorical_data['Inactive_Date'] = pd.to_datetime(categorical_data['Inactive_Date'])

# Create new column for Active_Dutation(months)
categorical_data['Active_Duration(months)'] = (categorical_data['Inactive_Date'] - categorical_data['Date_Joined']).dt.days // 30


# %% [markdown]
# Inactive_Duration(months) = Date_Resigned - Inactive_Date

# %%
# Convert 'Date_Resigned' 
categorical_data['Date_Resigned'] = pd.to_datetime(categorical_data['Date_Resigned'])


# Create new column for Active_Dutation(months)
categorical_data['Inactive_Duration(months)'] = (categorical_data['Date_Resigned'] - categorical_data['Inactive_Date']).dt.days // 30

# %%
categorical_data = categorical_data.drop(columns=['Inactive_Date'])
categorical_data = categorical_data.drop(columns=['Date_Joined'])
categorical_data = categorical_data.drop(columns=['Date_Resigned'])

# %%
categorical_data.head()

# %%
Marvellous = pd.concat([numeric_data, categorical_data], axis=1)

# %%
Marvellous.head()

# %% [markdown]
# Encoding

# %% [markdown]
# Binary Encoding

# %% [markdown]
# Binary Encoding for the feature 'Marital_Status'

# %%
mapping = {'Single': 0, 'Married': 1}
Marvellous['Marital_Status'].replace(mapping, inplace=True)

# %% [markdown]
# Binary Encoding for the feature 'Gender'

# %%
mapping = {'Female': 0, 'Male': 1}
Marvellous['Gender'].replace(mapping,inplace=True)

# %% [markdown]
# Binary Encoding for the feature 'Status'

# %%
mapping = {'Active': 0, 'Inactive': 1}
Marvellous['Status'].replace(mapping, inplace=True)

# %% [markdown]
# Binary Encoding for the feature 'Employement_Type'

# %%
mapping = {'Permanant': 0, 'Contarct Basis': 1}
Marvellous['Employment_Type'].replace(mapping, inplace=True)

# %% [markdown]
# Label Encoding

# %%
label_encoder = LabelEncoder()

# %%
Marvellous['Employment_Category'] = label_encoder.fit_transform(Marvellous['Employment_Category'])
Marvellous['Designation'] = label_encoder.fit_transform(Marvellous['Designation'])

# %%
Marvellous.head(15)

# %%
Marvellous.isnull().sum()

# %%
#create the csv file of the preprossesed data set of the employees.csv
Marvellous.to_csv('employee_preprocess_17.csv', index=False)

# %%
print("Successfully preprocessed 'employees.csv' and saved to 'employee_preprocess_17.csv'")


