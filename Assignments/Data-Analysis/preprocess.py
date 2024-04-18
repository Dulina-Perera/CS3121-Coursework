import warnings
warnings.filterwarnings("ignore")

import colorama
colorama.init(autoreset=True)

import polars as pl # type: ignore

from colorama import Fore, Style


def col_domain(df, col):
    print(f'{Fore.BLACK + col}:')
    print(*df[col].unique(), sep='\n', end='\n\n')


def cols_with_missing_vals(df):
    # Calculate the total number of rows in the dataframe.
    row_count = df.height

    # Calculate the count of null values and percentage of missing values for each column.
    for col in df.columns:
        missing_count = df[col].null_count()

        if missing_count > 0:
            missing_percentage = (missing_count / row_count) * 100
            
            print(Fore.GREEN + col, end=" ")
            print("has", end=" ")
            print(Fore.RED + str(missing_count), end=" ")
            print("missing values.", end=" ")
            print(Fore.RED + f"({missing_percentage:.2f}%)")


if __name__ == "__main__":
    df = pl.read_csv("data.csv")
    df.describe()

    # print(*df.columns, sep='\n', end='\n\n')
    # col_domain(df, 'Timestamp')
    # cols_with_missing_vals(df)
