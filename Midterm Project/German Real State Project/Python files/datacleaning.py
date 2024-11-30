import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self, data):
        """
        Initialize the DataCleaner with a DataFrame
        """
        self.data = data.copy()
        print("DataCleaner initialized with data of shape:", self.data.shape)
    
    def fill_categorical_nas(self, columns, fill_value='unknown', replacement_map=None):
        """
        Fill NaN values and optionally replace specific values in categorical columns
        """
        print(f"Filling NaNs in columns: {columns}")
        for col in columns:
            self.data[col] = self.data[col].fillna(fill_value)
            if replacement_map and col in replacement_map:
                self.data[col] = self.data[col].replace(replacement_map[col])
        print(f"Completed filling NaNs for columns: {columns}")
        return self
    
    def assign_top_k_categories(self, column, k):
        """
        Assign NaN values to top K categories proportionally
        """
        print(f"Assigning NaN values proportionally in '{column}' to top {k} categories")
        value_counts = self.data[column].value_counts(normalize=True)
        k = min(k, len(value_counts))
        top_k_categories = value_counts.head(k).index
        
        nan_indices = self.data[column][self.data[column].isna()].index
        if len(nan_indices) > 0:
            probabilities = value_counts.loc[top_k_categories].values
            probabilities = probabilities / probabilities.sum() 
            self.data.loc[nan_indices, column] = np.random.choice(
                top_k_categories, 
                size=len(nan_indices), 
                p=probabilities, 
                replace=True  
            )
        print(f"Completed assigning NaN values in '{column}'")
        return self
    
    def distribute_nan_proportionally(self, columns):
        """
        Distribute NaN values proportionally across existing categories for multiple columns
        """
        print(f"Distributing NaNs proportionally for columns: {columns}")
        for column in columns:
            value_counts = self.data[column].value_counts(normalize=True, dropna=False)
            nan_count = value_counts.get(np.nan, 0)
            non_nan_value_counts = value_counts.dropna()
            normalized_non_nan_counts = non_nan_value_counts / non_nan_value_counts.sum()
            if nan_count > 0:
                nan_indices = self.data[column][self.data[column].isna()].index
                num_nans = len(nan_indices)
                sampled_values = np.random.choice(
                    normalized_non_nan_counts.index, 
                    size=num_nans, 
                    p=normalized_non_nan_counts.values, 
                    replace=True
                )
                self.data.loc[nan_indices, column] = sampled_values
        print("Completed distributing NaNs proportionally")
        return self

    def fill_NAs_with_median(self, columns):
        """
        Fill NaN values in specified columns with their respective medians.
        
        Parameters:
        columns (list): List of column names to process
        
        Returns:
        self: Returns the instance for method chaining
        """
        for column in columns:
            if column in self.data.columns:
                median_value = self.data[column].median()
                print(f"Median value for '{column}': {median_value}")
                self.data[column].fillna(median_value, inplace=True)
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return self
    
    def drop_nas(self, columns):
        """
        Drop rows with NaN values in specified columns
        """
        print(f"Dropping rows with NaNs in columns: {columns}")
        initial_shape = self.data.shape
        self.data = self.data.dropna(subset=columns)
        print(f"Dropped rows with NaNs. Shape changed from {initial_shape} to {self.data.shape}")
        return self
    
    def drop_columns(self, columns):
        """
        Drop specified columns from the DataFrame
        """
        print(f"Dropping columns: {columns}")
        initial_shape = self.data.shape
        self.data.drop(columns, axis=1, inplace=True)
        print(f"Dropped columns. Shape changed from {initial_shape} to {self.data.shape}")
        return self
    
    def drop_columns_with_high_nulls(self, threshold=0.75):
        """
        Drop columns with more than the specified percentage of null values
        """
        initial_shape = self.data.shape
        print(f"Initial dataframe shape before dropping high-null columns: {initial_shape}")
        self.data = self.data.select_dtypes(include=['number']).dropna(axis=1, thresh=int(threshold * len(self.data)))
        print(f"Dropped high-null columns. Final shape: {self.data.shape}")
        return self
    
    def remove_outliers(self, columns):
        """
        Remove outliers from specified columns based on the IQR method.
        
        Parameters:
        columns (list): List of column names to process
        
        Returns:
        self: Returns the instance for method chaining
        """
        for column in columns:
            if column in self.data.columns:
                # Calculate Q1, Q3, and IQR
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Log the bounds for debugging
                print(f"Outlier bounds for '{column}': [{lower_bound}, {upper_bound}]")
                
                # Remove outliers
                initial_count = len(self.data)
                self.data = self.data[
                    (self.data[column] >= lower_bound) & 
                    (self.data[column] <= upper_bound)
                ]
                final_count = len(self.data)
                print(f"Removed {initial_count - final_count} outliers from '{column}'")
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return self
    
    def percent_of_outliers(self, columns):
        """
        Calculate the percentage of outliers in the specified columns based on the IQR method.
        
        Parameters:
        columns (list): List of column names to process
        
        Returns:
        dict: A dictionary with column names as keys and outlier percentages as values
        """
        outlier_percentages = {}
        for column in columns:
            if column in self.data.columns:
                # Calculate Q1, Q3, and IQR
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify the outliers
                outliers = self.data[
                    (self.data[column] < lower_bound) | 
                    (self.data[column] > upper_bound)
                ]
                
                # Calculate the percentage of outliers
                outlier_percentage = (len(outliers) / len(self.data)) * 100
                outlier_percentages[column] = outlier_percentage
                
                # Log the results
                print(f"Column '{column}': {outlier_percentage:.2f}% outliers")
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return outlier_percentages


    def categorize_columns(self, columns, bin_edges, category_names, drop_original=True):
        """
        Categorize multiple columns based on specified bins and labels.
        
        Parameters:
        columns (list): List of column names to categorize
        bin_edges (list): List of bin edges for categorization
        category_names (list): List of labels for the categories
        drop_original (bool): Whether to drop the original column after categorization (default: True)
        
        Returns:
        self: Returns the instance for method chaining
        """
        for column in columns:
            if column in self.data.columns:
                # Create a new column for the categorized data
                categorized_column_name = f"{column}_category"
                self.data[categorized_column_name] = pd.cut(
                    self.data[column],
                    bins=bin_edges,
                    labels=category_names,
                    right=False
                ).astype(object)  # Convert to object for easier handling
                
                print(f"Categorized '{column}' into '{categorized_column_name}' with bins {bin_edges} and labels {category_names}")
                
                # Drop the original column if requested
                if drop_original:
                    self.data.drop(columns=[column], inplace=True)
                    print(f"Dropped the original column '{column}'.")
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return self
    

    def categorize_with_bins(self, columns, bin_thresholds, category_labels, show_counts=False, drop_original=False):
        """
        Categorize multiple columns based on specified bin thresholds and labels.
        
        Parameters:
        columns (list): List of column names to categorize
        bin_thresholds (list): List of bin edges for categorization
        category_labels (list): List of labels for the categories
        show_counts (bool): Whether to display category counts for each new column (default: False)
        drop_original (bool): Whether to drop the original column after categorization (default: False)
        
        Returns:
        self: Returns the instance for method chaining
        """
        for column in columns:
            if column in self.data.columns:
                # Create a new column name for the categorized data
                categorized_column_name = f"{column}_category"
                self.data[categorized_column_name] = pd.cut(
                    self.data[column],
                    bins=bin_thresholds,
                    labels=category_labels
                )
                
                print(f"Categorized '{column}' into '{categorized_column_name}' using thresholds {bin_thresholds} and labels {category_labels}.")
                
                if show_counts:
                    counts = self.data[categorized_column_name].value_counts()
                    print(f"Category counts for '{categorized_column_name}':\n{counts}\n")
                
                # Drop the original column if requested
                if drop_original:
                    self.data.drop(columns=[column], inplace=True)
                    print(f"Dropped the original column '{column}'.")
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return self
    
    def process_columns_with_bins_and_fillna(
        self,
        columns,
        bin_thresholds=None,
        category_labels=None,
        remove_top_n=None,
        fill_na_with_median=False,
        show_counts=False,
        drop_original=False
    ):
        """
        Process multiple columns by:
        1. Categorizing them based on bin thresholds and labels.
        2. Removing top N maximum values (optional).
        3. Filling NaN values with the column median (optional).

        Parameters:
        columns (list): List of column names to process.
        bin_thresholds (list): List of bin edges for categorization (optional).
        category_labels (list): List of labels for the categories (optional).
        remove_top_n (int): Number of top maximum values to remove (optional).
        fill_na_with_median (bool): Whether to fill NaN values with median (default: False).
        show_counts (bool): Whether to display category counts for each new column (default: False).
        drop_original (bool): Whether to drop the original column after categorization (default: False).

        Returns:
        self: Returns the instance for method chaining.
        """
        for column in columns:
            if column in self.data.columns:
                print(f"Processing column: '{column}'")
                
                # Step 1: Categorize column (if thresholds and labels are provided)
                if bin_thresholds and category_labels:
                    categorized_column_name = f"{column}_category"
                    self.data[categorized_column_name] = pd.cut(
                        self.data[column],
                        bins=bin_thresholds,
                        labels=category_labels
                    )
                    print(f"Categorized '{column}' into '{categorized_column_name}' using thresholds {bin_thresholds} and labels {category_labels}.")
                    
                    if show_counts:
                        counts = self.data[categorized_column_name].value_counts()
                        print(f"Category counts for '{categorized_column_name}':\n{counts}\n")
                    
                    # Optionally drop the original column
                    if drop_original:
                        self.data.drop(columns=[column], inplace=True)
                        print(f"Dropped the original column '{column}'.")
                
                # Step 2: Remove top N maximum values
                if remove_top_n:
                    top_n_max_values = self.data[column].nlargest(remove_top_n)
                    self.data = self.data[~self.data[column].isin(top_n_max_values)]
                    print(f"Removed top {remove_top_n} maximum values from '{column}'.")
                
                # Step 3: Fill NaN values with median
                if fill_na_with_median:
                    median_value = self.data[column].median()
                    self.data[column].fillna(median_value, inplace=True)
                    print(f"Filled NaN values in '{column}' with median: {median_value}.")
            else:
                print(f"Column '{column}' does not exist in the dataset.")
        
        return self

    
    def get_cleaned_data(self):
        """
        Return the cleaned DataFrame
        """
        print("Returning the cleaned data.")
        return self.data

# Usage
def get_data(name):
    current_dir = os.getcwd()
    data_path = "Midterm Project/German Real State Project/Data"
    file_name = f"{name}.csv"
    file_path = os.path.join(current_dir, data_path, file_name)
    return pd.read_csv(file_path)

# Load data
data = get_data('immo_data')
# Initialize the cleaner
cleaner = DataCleaner(data)


# Cleaning pipeline with print statements
cleaned_data = (cleaner
    .fill_categorical_nas(
        ['petsAllowed', 'interiorQual', 'condition', 'telekomTvOffer'], 
        replacement_map={'telekomTvOffer': {'NONE': 'unknown'}}
    )
    .assign_top_k_categories('firingTypes', k=4)
    .distribute_nan_proportionally(['heatingType', 'typeOfFlat'])
    .drop_nas(['heatingType'])
    .drop_columns([
        'energyEfficiencyClass', 
        'houseNumber', 
        'streetPlain', 
        'facilities', 
        'description', 
        'date', 
        'street', 
        'geo_bln', 
        'geo_krs', 
        'noRoomsRange'
    ])
    .drop_columns_with_high_nulls()
    .fill_NAs_with_median(['serviceCharge','priceTrend','telekomUploadSpeed'])
    .remove_outliers(['yearConstructed','yearConstructedRange','floor'])
    .categorize_columns(
        columns=['yearConstructed'],
        bin_edges=[1881, 1952, 1973, 1997, 2030],
        category_names=['1881-1952', '1952-1973', '1973-1997', '1997-2030'],
        drop_original=True
    )
    .categorize_with_bins(
        columns=['floor'], 
        bin_thresholds=[-float('inf'), 0, 3.0, 5.0, 45.0, float('inf')], 
        category_labels=['ground_floor_and_below', 'floor_0_3', 'floor_4_5', 'floor_6_45', 'floor_above_45'], 
        show_counts=True, 
        drop_original=True
    )
    .process_columns_with_bins_and_fillna(
        columns=['totalRent','baseRent','livingSpace','noRooms'],
        remove_top_n=10,
        fill_na_with_median=True
    )
    .get_cleaned_data()
)
