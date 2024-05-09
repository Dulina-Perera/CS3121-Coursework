# %%
import pandas as pd # type: ignore

from lifelines import KaplanMeierFitter # type: ignore
from lifelines.statistics import logrank_test # type: ignore
from matplotlib import pyplot as plt # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

# %%
class SalaryAnalyzer:
    def __init__(self, employees: pd.DataFrame, salaries: pd.DataFrame, salary_dict: pd.DataFrame):
        self._employees = employees.copy(deep=True)
        self._salaries = salaries.copy(deep=True)
        self._salary_dict = salary_dict.copy(deep=True)


    @property
    def employees(self):
        return self._employees


    @property
    def salaries(self):
        return self._salaries
    

    @property
    def salary_dict(self):
        return self._salary_dict


    @employees.setter
    def employees(self, employees):
        self._employees = employees


    @salaries.setter
    def salaries(self, salaries):
        self._salaries = salaries


    @salary_dict.setter
    def salary_dict(self, salary_dict):
        self._salary_dict = salary_dict


    def preprocess(self):
        self._rename_columns()
        
        self._merge_columns('Accomadation Allowance', 'Accommodation Allowance', 'Accomadation Allowance')
        self._merge_columns('Additional Allowance_0', 'Additional Allowance_2', 'Additional Allowance')
        self._merge_columns('Attendance Allowance_0', 'Attendance Allowance_2', 'Attendance Allowance')
        self._merge_columns('Basic Rate_2', 'Basic Rate_3', 'Basic Rate')
        self._merge_columns('Basic Salary_0', 'Basic Salary_2', 'Basic Salary')
        self._merge_columns('Hard Ship Allowance_0', 'Hard Ship Allowance_2', 'Hard Ship Allowance')
        self._merge_columns('OT Amount 1.5_0', 'OT Amount 1.5_2', 'OT Amount 1.5')
        self._merge_columns('Salary Arrears_0', 'Salary Arrears_2', 'Salary Arrears')
        self._merge_columns('Total Earnings_0', 'Total Earnings_2', 'Total Earnings')
        self._validate_against_salary_dict()

        self.employees.sort_values(by=['Employee_No'], inplace=True)
        self.salaries.sort_values(by=['Employee_No', 'Year', 'Month'], inplace=True)

        self._calculate_days_employed('2023-06-01')
        self._calculate_resign_status()


    def calculate_slope(self, column: str):
        slope_data = {}
        for employee, agg in self.salaries.groupby('Employee_No'):
            x_values = []
            y_values = []
            for index, row in agg.iterrows():
                x_values.append([index])  # Index as x-value
                y_values.append(row[column])  # Column value as y-value

            model = LinearRegression().fit(x_values, y_values)
            slope_data[employee] = model.coef_[0]
        
        self.employees[f'{column}_Slope'] = self.employees['Employee_No'].map(pd.DataFrame(slope_data.items(), columns=['Employee_No', f'{column}_Slope']).set_index('Employee_No')[f'{column}_Slope'])
        self.employees[f'{column}_Slope'] = ['Decreasing' if slope < 0 else 'Not Decreasing' for slope in self.employees[f'{column}_Slope']]


    def perform_logrank_test(self, category_column, event_column='Resign_Status', time_column='Days_Employed'):
        decreasing = self.employees[self.employees[category_column] == 'Decreasing']
        not_decreasing = self.employees[self.employees[category_column] == 'Not Decreasing']

        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))

        kmf.fit(decreasing[time_column], event_observed=decreasing[event_column], label='Decreasing')
        kmf.plot()

        kmf.fit(not_decreasing[time_column], event_observed=not_decreasing[event_column], label='Not Decreasing')
        kmf.plot()

        plt.title(f'Kaplan-Meier Curves for {category_column}')
        plt.xlabel(time_column)
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.show()
            
        # Perform log-rank test.
        results = logrank_test(durations_A=decreasing[time_column], durations_B=not_decreasing[time_column], event_observed_A=decreasing[event_column], event_observed_B=not_decreasing[event_column])

        # Print results.
        results.print_summary(decimals=4)


    def _rename_columns(self):
        self.salaries.rename(columns={'year': 'Year'}, inplace=True)
        self.salaries.rename(columns={'month': 'Month'}, inplace=True)
        self.salaries.rename(columns={'SiteNo': 'Site_No'}, inplace=True)
        self.salaries.rename(columns={'Hard ship Allowance Rate': 'Hard Ship Allowance Rate'}, inplace=True)

    
    def _merge_columns(self, column1: str, column2: str, new_column: str):
        filtered: pd.DataFrame = self.salaries[(self.salaries[column1] != 0) & (self.salaries[column2] != 0)]

        if filtered.empty:
            self.salaries[new_column] = self.salaries[column1] + self.salaries[column2]
            self.salaries.drop([column1, column2], axis=1, inplace=True)
        else:
            print(f"There are columns where both '{column1}' and '{column2}' have non-zero values.")


    def _validate_against_salary_dict(self):
        salary_components_dict: set = set.union(*(set(self.salary_dict[col].unique()) for col in self.salary_dict.columns))
        salary_components_salaries: set = set(self.salaries.columns.to_list()[5:])

        components_only_in_dict: set = salary_components_dict - salary_components_salaries
        components_only_in_salaries: set = salary_components_salaries - salary_components_dict

        print("Components only present in salary_dict:", components_only_in_dict)
        print("Components only present in salaries:", components_only_in_salaries)


    def _calculate_days_employed(self, today: str):
        # Convert Date_Joined and Date_Resigned to datetime objects.
        self.employees['Date_Joined'] = pd.to_datetime(self.employees['Date_Joined'])
        self.employees['Date_Resigned'] = pd.to_datetime(self.employees['Date_Resigned'])
        
        # Calculate the difference between Date_Joined and the specified date.
        today: pd.Timestamp = pd.Timestamp(today)
        difference = (today - self.employees['Date_Joined']).dt.days
        
        # Calculate Days_Employed.
        self.employees['Days_Employed'] = (self.employees['Date_Resigned'] - self.employees['Date_Joined']).dt.days
        self.employees['Days_Employed'].fillna(difference, inplace=True)
        
        self.employees['Days_Employed'] = self.employees['Days_Employed'].astype('Int64')


    def _calculate_resign_status(self):
        self.employees['Resign_Status'] = self.employees['Date_Resigned'].notnull().astype(int)


# %%
