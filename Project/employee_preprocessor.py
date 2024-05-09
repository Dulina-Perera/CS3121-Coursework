# %%
import numpy as np # type: ignore
import pandas as pd # type: ignore
import re

# %%
class EmployeePreprocessor:
    def __init__(self, employees: pd.DataFrame):
        self._employees = employees.copy(deep=True)


    @property
    def employees(self):
        return self._employees


    @employees.setter
    def employees(self, employees):
        self._employees = employees


    def preprocess(self):
        # The column 'Employee_Code' serves no purpose.
        self.employees.drop(columns=['Employee_Code'], inplace=True)

        # Handle different missing value representations.
        self.employees.loc[self.employees['Date_Resigned'] == '0000-00-00', 'Date_Resigned'] = np.nan
        self.employees.loc[self.employees['Date_Resigned'] == '\\N', 'Date_Resigned'] = np.nan

        self.employees.loc[self.employees['Inactive_Date'] == '0000-00-00', 'Inactive_Date'] = np.nan
        self.employees.loc[self.employees['Inactive_Date'] == '\\N', 'Inactive_Date'] = np.nan

        self.employees.loc[self.employees['Reporting_emp_1'] == "\\N", 'Reporting_emp_1'] = np.nan
        self.employees.loc[self.employees['Reporting_emp_2'] == "\\N", 'Reporting_emp_2'] = np.nan

        self.employees.loc[self.employees['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = np.nan

        # Drop columns with more than 95% missing values.
        self.employees.dropna(thresh=0.05 * self.employees.shape[0], axis=1, inplace=True)

        # Drop one of the 'Religion_ID' or 'Religion' columns since they are duplicates.
        if self.employees['Religion_ID'].nunique() == self.employees['Religion'].nunique() and \
        (self.employees.groupby('Religion_ID')['Religion'].nunique() == 1).all():
            self.employees.drop(columns=['Religion_ID'], inplace=True)

        # Drop one of the 'Designation_ID' or 'Designation' columns since they are duplicates.
        if self.employees['Designation_ID'].nunique() == self.employees['Designation'].nunique() and \
        (self.employees.groupby('Designation_ID')['Designation'].nunique() == 1).all():
            self.employees.drop(columns=['Designation_ID'], inplace=True)

        # The columns 'Title' and 'Gender' can be inferred from the 'Name' column.
        self._update_title_and_gender_on_name()

        # Nullify conflicting records in the 'Title' and 'Gender' columns.
        self._fill_conflicting_title_and_gender()

        # Extract the first name and last name from the 'Name' column.
        self._extract_fname_and_lname()

        # Resolve inconsistencies between the 'Title' and 'Marital_Status' columns.
        self.employees.loc[(self.employees['Title'] == 'Mrs') & (self.employees['Marital_Status'].isna()), 'Marital_Status'] = 'Married'
        self.employees.loc[(self.employees['Title'] == 'Ms') & (self.employees['Marital_Status'].isna()), 'Title'] = np.nan
        self.employees.loc[(self.employees['Title'] == 'Ms') & (self.employees['Marital_Status'] == 'Single'), 'Title'] = 'Miss'
        self.employees.loc[(self.employees['Title'] == 'Ms') & (self.employees['Marital_Status'] == 'Married'), 'Title'] = 'Mrs'

        # Using 


    def _update_title_and_gender_on_name(self):
        self.employees.loc[self.employees['Name'].str.startswith('Mr'), ['Title', 'Gender']] = 'Mr', 'Male'
        self.employees.loc[self.employees['Name'].str.startswith('Ms'), ['Title', 'Gender']] = 'Ms', 'Female'
        self.employees.loc[self.employees['Name'].str.startswith('Miss'), ['Title', 'Gender']] = 'Miss', 'Female'
        self.employees.loc[self.employees['Name'].str.startswith('Mrs'), ['Title', 'Gender']] = 'Mrs', 'Female'


    def _fill_conflicting_title_and_gender(self):
        self.employees.loc[(self.employees['Title'] == 'Miss') & (self.employees['Gender'] == 'Male'), ['Title', 'Gender']] = np.nan, np.nan
        self.employees.loc[(self.employees['Title'] == 'Ms') & (self.employees['Gender'] == 'Male'), ['Title', 'Gender']] = np.nan, np.nan
        self.employees.loc[(self.employees['Title'] == 'Mrs') & (self.employees['Gender'] == 'Male'), ['Title', 'Gender']] = np.nan, np.nan
        self.employees.loc[(self.employees['Title'] == 'Mr') & (self.employees['Gender'] == 'Female'), ['Title', 'Gender']] = np.nan, np.nan

    
    def _extract_fname_and_lname(self):
        regex: str = r'^(?:(?:Mr|Mrs|Miss|Ms|Dr|Prof)?\s+)?(?P<First_Name>\S+)\s*(?P<Last_Name>\S+)?\s*(?:DDS|DVM|I|II|III|IV|Jr|MD|PhD|Sr|V)?$'
        self.employees[['First_Name', 'Last_Name']] = self.employees['Name'].str.extract(regex)[['First_Name', 'Last_Name']]

        self.employees.drop(columns=['Name'], inplace=True)

# %%
