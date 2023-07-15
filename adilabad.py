import pandas as pd

def filter_csv_by_name(csv_file, name_column, specified_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Filter rows based on the specified name
    filtered_df = df[df[name_column] == specified_name]

    # Save the filtered DataFrame back to a CSV file
    filtered_df.to_csv(csv_file, index=False)

# Specify the name of the CSV file
csv_file = 'Combined.csv'

# Specify the name of the column containing the names
name_column = 'District'

# Specify the name to filter for
specified_name = 'Adilabad'

# Call the function to filter the CSV file
filter_csv_by_name(csv_file, name_column, specified_name)