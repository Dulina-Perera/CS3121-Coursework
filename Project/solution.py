# %%
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import warnings

from IPython.display import display # type: ignore

from salary_analyzer import SalaryAnalyzer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings("ignore")

# %%
# Load the data.
attendance = pd.read_csv('Resources/attendance.csv')
employees = pd.read_csv('Resources/employees.csv')
holidays = pd.read_csv('Resources/holidays.csv')
leaves = pd.read_csv('Resources/leaves.csv')
salary_dict = pd.read_csv('Resources/salary_dictionary.csv')
salaries = pd.read_csv('Resources/salary.csv')

# salary_dict = pd.read_csv('Resources/salary_dictionary.csv')

# %%
# Get an overview of the data.
# display(attendance.info(), attendance.head())
# display(employees.info(), employees.head())
# display(holidays.info(), holidays.head())
# display(leaves.info(), leaves.head())
# display(salaries.info(), salaries.head())

# display(salary_dict.info(), salary_dict.head())

# %% [markdown]
# ## Handling Missing Values

# %%
def columns_with_missing_values(df):
    """
    Returns a DataFrame with columns that have missing values and the count of missing values in each column.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: A DataFrame with columns that have missing values and the count of missing values in each column.
    """

    # Calculate the count of missing values in each column.
    missing = df.isna().sum()

    # Filter out columns with no missing values.
    missing = missing[missing > 0]

    # Set the display option to show all rows.
    pd.set_option('display.max_rows', None)

    # Display the DataFrame with columns that have missing values.
    display(missing)

    # Reset the display option to show a maximum of 10 rows.
    pd.set_option('display.max_rows', 10)


def display_unique_values(df, columns):
    """
    Function to display the unique values in each column of a DataFrame.

    Args:
    - df: DataFrame: The pandas DataFrame to analyze.
    - columns: list: A list of column names in the DataFrame for which unique values are to be displayed.

    Returns:
    - None: This function does not return any value. It prints the unique values for each specified column.

    Example:
    Suppose we have a DataFrame df with columns 'A', 'B', and 'C'. To display the unique values in columns 'A' and 'B',
    we can call display_unique_values(df, ['A', 'B']).
    """
    for col in columns:
        # Find the unique values in the specified column.
        unique_values = df[col].unique()
        # Print the unique values for the current column.
        print(f"Unique values in column '{col}': {unique_values}", end="\n\n")

# %% [markdown]
# 1. **Attendance Dataset**

# %%
columns_with_missing_values(attendance)

# %%
for col in attendance.columns:
    display(attendance[col].value_counts())

# %% [markdown]
# 2. **Employees Dataset**

# %%
def handle_employees():
    employees.loc[employees['Date_Resigned'] == '0000-00-00', 'Date_Resigned'] = np.nan
    employees.loc[employees['Date_Resigned'] == '\\N', 'Date_Resigned'] = np.nan

    employees.loc[employees['Inactive_Date'] == '0000-00-00', 'Inactive_Date'] = np.nan
    employees.loc[employees['Inactive_Date'] == '\\N', 'Inactive_Date'] = np.nan

    employees.loc[employees['Reporting_emp_1'] == "\\N", 'Reporting_emp_1'] = np.nan

    employees.drop('Reporting_emp_2', axis=1, inplace=True)

    employees.loc[employees['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = np.nan

# %%
display(employees.describe())

columns_with_missing_values(employees)
# display_unique_values(employees, ['Marital_Status', 'Date_Resigned', 'Inactive_Date', 'Reporting_emp_1', 'Reporting_emp_2', 'Year_of_Birth'])

# %% [markdown]
# ### Missing values are defined by the business context.

# > - The column 'Marital_Status' must have a value of 'Single' or 'Married'.
# > - Thus, the NaN values in the 'Marital_Status' column are missing values.

#

# > - The columns 'Date_Resigned', 'Status', and 'Inactive_Date' are related by two rules.
# >   1. If an employee is active, then 'Date_Resigned' and 'Inactive_Date' should have a value of '\N' or '0000-00-00'.
# >   2. If an employee is inactive, then 'Inactive_Date' should have a valid date and 'Date_Resigned' should have a value of '\N' or '0000-00-00'.
# > - Breaking any of the above rules result in missing values in the corresponding columns.

#

# > - The column 'Year_of_Birth' has a value of "'0000'".
# > - This value is not a valid year of birth and should be treated as a missing value.

#

# > - The column 'Reporting_emp_1' has a value of '\N'.
# > - Not all employees have a reporting employee, so this value is not a missing value.

# %%
# Check if all active employees have a 'Date_Resigned' value of '\N' or '0000-00-00'. - Yes
filtered = employees[employees['Status'] == 'Active']['Date_Resigned']
# display(filtered.value_counts())

# Check if all active employees have a 'Inactive_Date' value of '\N' or '0000-00-00'. - Yes
filtered = employees[employees['Status'] == 'Active']['Inactive_Date']
# display(filtered.value_counts())

# Check if all inactive employees have a valid 'Inactive_Date'. - Yes
filtered = employees[(employees['Status'] == 'Inactive') & ((employees['Inactive_Date'] == '0000-00-00') | (employees['Inactive_Date'] == '\\N'))]['Inactive_Date']
# display(filtered.value_counts())

# Check if all inactive employees have resigned. - No
filtered = employees[employees['Status'] == 'Inactive']['Date_Resigned']
# display(filtered.value_counts())

# %%
handle_employees()

# %% [markdown]
# ## Effect of Salary on Employee Attrition

# %%
analyser = SalaryAnalyzer(employees, salaries, salary_dict)

analyser.preprocess()

# %%
pd.set_option('display.max_rows', None)
display(analyser.salary_dict)

# %%
analyser.perform_logrank_test('Basic Salary')

# %%
