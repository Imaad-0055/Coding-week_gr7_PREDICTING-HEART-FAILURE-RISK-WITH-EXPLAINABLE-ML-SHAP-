# Test of data_processing

import pandas as pd
import numpy as np

# Read the dataset
d = pd.read_csv("heart_failure_clinical_records_dataset.csv")

def optimize_memory(df):
    """
    Optimize memory usage of a pandas DataFrame by downcasting numeric data types.

    Parameters:
    -----------
    df : pandas.DataFrame
    Returns:
    --------
    pandas.DataFrame
        A memory-optimized copy of the input DataFrame
    """
    # Create a copy of the dataframe to avoid modifying the original
    result = df.copy()

    # Memory usage before optimization
    start_memory = result.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before optimization: {start_memory:.2f} MB")

    # Optimize numeric columns
    for col in result.columns:
        col_type = result[col].dtype

        # Process numeric columns
        if pd.api.types.is_numeric_dtype(col_type):

            # Integers
            if pd.api.types.is_integer_dtype(col_type):
                # Get min and max values to determine the smallest possible type
                c_min = result[col].min()
                c_max = result[col].max()

                # Determine the best integer type based on min and max values
                if c_min >= 0:  # For unsigned integers
                    if c_max < 255:
                        result[col] = result[col].astype(np.uint8)
                    elif c_max < 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        result[col] = result[col].astype(np.uint32)
                    else:
                        result[col] = result[col].astype(np.uint64)
                else:  # For signed integers
                    if c_min > -128 and c_max < 127:
                        result[col] = result[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767:
                        result[col] = result[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647:
                        result[col] = result[col].astype(np.int32)
                    else:
                        result[col] = result[col].astype(np.int64)

            # Floats
            elif pd.api.types.is_float_dtype(col_type):
                # Downcast to float32 if possible
                c_min = result[col].min()
                c_max = result[col].max()

                # Check if the float32 range is sufficient
                # (approximate range: -3.4e38 to 3.4e38)
                if c_min > -3.4e38 and c_max < 3.4e38:
                    result[col] = result[col].astype(np.float32)
                else:
                    result[col] = result[col].astype(np.float64)

        # For object columns, convert to category if beneficial
        elif col_type == 'object':
            # Calculate the ratio of unique values to total values
            unique_ratio = result[col].nunique() / len(result)

            # If the ratio is small, it's beneficial to use categorical
            if unique_ratio < 0.5:  # This threshold can be adjusted
                result[col] = result[col].astype('category')

    # Memory usage after optimization
    end_memory = result.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage after optimization: {end_memory:.2f} MB")
    print(f"Memory reduced by: {100 * (start_memory - end_memory) / start_memory:.2f}%")

    return result

# Run the memory optimization function
optimized_df = optimize_memory(d)
