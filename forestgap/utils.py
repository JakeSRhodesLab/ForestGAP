import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#----------------------------------------------------------------------------#
#<><><><><><><><><><><><><> Missing Data Transform <><><><><><><><><><><><><>#
#----------------------------------------------------------------------------#

def missing_prep_transform(train_data, test_data=None, continuous_transform='standardize', categorical_transform='encode'):
    """
    Preprocesses a DataFrame (or numpy array) by applying transformations
    to continuous and categorical features.

    Parameters:
    - train_data (pd.DataFrame or np.ndarray): The input data for training.
    - test_data (pd.DataFrame or np.ndarray): An optional test dataset transformed
                                               using the scaler fit on training data.
    - continuous_transform (str): 'standardize' or 'normalize'. Default is 'standardize'.
    - categorical_transform (str): 'encode' or 'onehot'. Default is 'encode'.

    Returns:
    - pd.DataFrame or np.ndarray (matching input type), or a tuple of two if
      test_data is provided.
    """
    _train_is_array = isinstance(train_data, np.ndarray)
    _test_is_array = isinstance(test_data, np.ndarray)
    if _train_is_array:
        train_data = pd.DataFrame(train_data.astype(float))
    if _test_is_array:
        test_data = pd.DataFrame(test_data.astype(float))
    # Identify continuous and other features (non-continuous)
    continuous_features = train_data.select_dtypes(include='float').columns
    other_features = train_data.select_dtypes(exclude='float').columns

    # Initialize scalers
    scaler = None
    if continuous_transform == 'standardize':
        scaler = StandardScaler()
    elif continuous_transform == 'normalize':
        scaler = MinMaxScaler()
    else:
        raise ValueError("continuous_transform must be 'standardize' or 'normalize'")

    # Process continuous features in training data
    if len(continuous_features) > 0:
        train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])
        if test_data is not None:
            test_data[continuous_features] = scaler.transform(test_data[continuous_features])

    # Process categorical features
    if len(other_features) > 0:
        if categorical_transform == 'encode':
            # Integer encoding for categorical features
            for feature in other_features:
                train_cat = pd.Categorical(train_data[feature])
                categories = train_cat.categories
                train_data[feature] = train_cat.codes
                if test_data is not None:
                    # Ensure test data categories align with training data
                    test_data[feature] = pd.Categorical(test_data[feature],
                                                        categories=categories).codes
        elif categorical_transform == 'onehot':
            # One-hot encoding for categorical features
            train_data = pd.get_dummies(train_data, columns=other_features)
            if test_data is not None:
                test_data = pd.get_dummies(test_data, columns=other_features)
                # Align test_data columns with train_data columns
                test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
        else:
            raise ValueError("categorical_transform must be 'encode' or 'onehot'")

    if _train_is_array:
        train_data = train_data.to_numpy(dtype=float, na_value=np.nan)
    if _test_is_array and test_data is not None:
        test_data = test_data.to_numpy(dtype=float, na_value=np.nan)
    return (train_data, test_data) if test_data is not None else train_data




#-----------------------------------------------------------------#
#<><><><><><><><><><><><><> Remove MCAR <><><><><><><><><><><><><>#
#-----------------------------------------------------------------#

def remove_mcar(df, pct=0.10, random_state=42):
    
    """
    Removes a specified percentage of values from each column in the DataFrame 
    randomly, simulating a Missing Completely at Random (MCAR) scenario.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame from which to remove values.
    pct : float, optional, default=0.10
        The percentage of values to remove from each column. Should be a value 
        between 0 and 1.
    random_state : int, optional, default=42
        The seed for the random number generator to ensure reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the specified percentage of values removed from each column.

    Notes
    -----
    The function masks values in different ways depending on the data type of each column:
    - Integer columns are replaced with `pd.NA` (nullable integer type).
    - Numeric columns (float types) are replaced with `np.nan`.
    - Categorical columns are replaced with `pd.NA`.
    - Object (string) columns are replaced with `None`.
    - DateTime columns are replaced with `pd.NaT`.
    """
    
    _input_is_array = isinstance(df, np.ndarray)
    if _input_is_array:
        df = pd.DataFrame(df.astype(float))

    np.random.seed(random_state)  # Set the seed for reproducibility
    
    def mask_column(column, pct):
        """Helper function to mask a specified percentage of values in a column."""
        # Create a random mask where a specified percentage is masked
        mask = np.random.rand(len(column)) >= pct
        
        # Mask values based on column type
        if pd.api.types.is_integer_dtype(column):
            return column.astype("Int64").where(mask, pd.NA)  # Convert to nullable Int64
        elif pd.api.types.is_numeric_dtype(column):
            return column.where(mask, np.nan)  # Use np.nan for floats
        elif isinstance(column.dtype, pd.CategoricalDtype) or (hasattr(pd.api.types, 'is_categorical_dtype') and pd.api.types.is_categorical_dtype(column)):
            return column.where(mask, pd.NA)  # Use pd.NA for categorical
        elif pd.api.types.is_object_dtype(column):
            return column.where(mask, None)  # Use None for objects
        elif pd.api.types.is_datetime64_any_dtype(column):
            return column.where(mask, pd.NaT)  # Use pd.NaT for datetime
        else:
            return column.where(mask)  # Fallback for unknown dtypes
    
    # Apply the masking function to each column in the DataFrame
    result = df.apply(lambda col: mask_column(col, pct))
    if _input_is_array:
        return result.to_numpy(dtype=float, na_value=np.nan)
    return result




# TODO: Test and update as needed
def remove_mnar(X, q, p, columns=None, random_state=None):
    """
    Introduce MNAR (Missing Not At Random) values into a DataFrame.

    For each selected column, values above the `q`-th quantile are randomly set 
    to NA with probability `p`, preserving original dtypes.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    q : float
        Quantile threshold (0 < q < 1), e.g., 0.8 for the 80th percentile.
    p : float
        Probability of setting a value above the quantile to NA (0 ≤ p ≤ 1).
    columns : list of str, optional
        Subset of columns to apply MNAR to. If None, all numeric columns are used.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with MNAR values introduced.
    """
    if not (0 < q < 1):
        raise ValueError("`q` must be between 0 and 1 (exclusive).")
    if not (0 <= p <= 1):
        raise ValueError("`p` must be between 0 and 1 (inclusive).")

    _input_is_array = isinstance(X, np.ndarray)
    if _input_is_array:
        X = pd.DataFrame(X.astype(float))
        # ndarray columns are integer indices; pass through as-is

    X = X.copy()
    rng = np.random.default_rng(random_state)

    if columns is None:
        columns = X.select_dtypes(include=np.number).columns
    else:
        missing_cols = [col for col in columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    for col in columns:
        # Convert to nullable dtype to preserve dtype with missing values
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype("Int64")
        elif pd.api.types.is_float_dtype(X[col]):
            X[col] = X[col].astype("Float64")

        quantile = np.quantile(X[col].dropna(), q)
        above_quant = X[col] > quantile
        sample = rng.binomial(1, p, size=X.shape[0])
        mask = above_quant & (sample == 1)
        X.loc[mask, col] = pd.NA

    if _input_is_array:
        return X.to_numpy(dtype=float, na_value=np.nan)
    return X



def get_correlated_features(df, threshold = 0.5, return_correlation = False):
    """
    Returns a list of unique column pairs in the DataFrame that have
    a correlation above the specified threshold (in absolute value).
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The correlation threshold (default is 0.8).
    
    Returns:
        List of tuples: [(col1, col2, correlation), ...]
    """
    corr_matrix = df.corr()
    pairs = [
        (col1, col2, corr_matrix.loc[col1, col2])
        for col1 in corr_matrix.columns
        for col2 in corr_matrix.columns
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold
    ]
    # Deduplicate pairs by sorting column names and using a set

    if return_correlation:
        unique_pairs = list({
            tuple(sorted((col1, col2))) + (corr,)
            for col1, col2, corr in pairs
        })

    else:
        unique_pairs = list({
            tuple(sorted((col1, col2)))
            for col1, col2, corr in pairs
        })
    return unique_pairs




# Review; not yet tested. Make a way to autoselect targets and conditionals based on correlation between them.


def remove_mar(X, q, p, targets=None, conditionals=None, correlation_threshold=0.5, random_state=None):
    """
    Introduce MAR (Missing At Random) values into a DataFrame.
    
    For each target column, values are set to NA with probability `p`,
    conditioned on whether the value in a corresponding column exceeds the `q`-th quantile.
    Original data types are preserved using pandas' nullable dtypes (e.g., Int64, Float64).

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    q : float
        Quantile threshold (0 < q < 1), used on conditioning columns.
    p : float
        Probability of missingness (0 ≤ p ≤ 1).
    targets : list of str, optional
        Columns in which to introduce missingness. If None, use all numeric columns.
    conditionals : list of str, optional
        Columns used to condition the missingness. If None, use the same as targets.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with MAR values introduced, preserving original dtype semantics.
    """
    if not (0 < q < 1):
        raise ValueError("`q` must be between 0 and 1 (exclusive).")
    if not (0 <= p <= 1):
        raise ValueError("`p` must be between 0 and 1 (inclusive).")

    _input_is_array = isinstance(X, np.ndarray)
    if _input_is_array:
        X = pd.DataFrame(X.astype(float))

    X = X.copy()
    rng = np.random.default_rng(random_state)

    if targets is None or conditionals is None:
        corr_features = get_correlated_features(X, threshold=correlation_threshold)
        if not corr_features:
            return X.to_numpy(dtype=float, na_value=np.nan) if _input_is_array else X
        targets, conditionals = zip(*corr_features)

    for target_col, cond_col in zip(targets, conditionals):
        if target_col == cond_col:
            continue
        if target_col not in X.columns or cond_col not in X.columns:
            raise ValueError(f"Column '{target_col}' or '{cond_col}' not in DataFrame.")

        try:
            # Promote to nullable dtype if needed
            if pd.api.types.is_integer_dtype(X[target_col]):
                X[target_col] = X[target_col].astype("Int64")
            elif pd.api.types.is_float_dtype(X[target_col]):
                X[target_col] = X[target_col].astype("Float64")

            threshold = np.quantile(X[cond_col].dropna(), q)
            condition = X[cond_col] > threshold
            sample = rng.binomial(1, p, size=X.shape[0])
            mask = condition & (sample == 1)

            X.loc[mask, target_col] = pd.NA

        except Exception as e:
            print(f"Error processing columns {target_col} and {cond_col}: {e}")
            continue

    if _input_is_array:
        return X.to_numpy(dtype=float, na_value=np.nan)
    return X


#----------------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><> Get Missing Dataframe <><><><><><><><><><><><><><><#
#----------------------------------------------------------------------------------#


def train_test(df, random_state=42):
    """
    Only currently set to work if the df has labels in the first column.
    """
    df = df.copy()
    label_column = df.columns[0]

    y = df.pop(label_column)

    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=random_state)

    return x_train, x_test, y_train, y_test



def apply_missingness(X, mechanism='mcar', percentage=0.10, random_state=42, **mechanism_kwargs):
    """
    Apply a chosen missingness mechanism using a unified interface.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input features.
    mechanism : str, default='mcar'
        One of {'mcar', 'mnar', 'mar'}.
    percentage : float, default=0.10
        Missingness level in [0, 1]. Used as:
        - `pct` for MCAR
        - default `p` for MNAR/MAR unless `p` is explicitly provided
    random_state : int, default=42
        Seed for reproducibility.
    **mechanism_kwargs : dict
        Additional mechanism-specific parameters:
        - MNAR: q, p, columns
        - MAR: q, p, targets, conditionals, correlation_threshold
    """
    if not (0 <= percentage <= 1):
        raise ValueError("`percentage` must be between 0 and 1 (inclusive).")

    mechanism = mechanism.lower()

    if mechanism == 'mcar':
        if mechanism_kwargs:
            raise ValueError(f"Unexpected kwargs for mcar: {sorted(mechanism_kwargs.keys())}")
        return remove_mcar(X, pct=percentage, random_state=random_state)

    if mechanism == 'mnar':
        q = mechanism_kwargs.pop('q', 0.8)
        p = mechanism_kwargs.pop('p', percentage)
        columns = mechanism_kwargs.pop('columns', None)
        if mechanism_kwargs:
            raise ValueError(f"Unexpected kwargs for mnar: {sorted(mechanism_kwargs.keys())}")
        return remove_mnar(X, q=q, p=p, columns=columns, random_state=random_state)

    if mechanism == 'mar':
        q = mechanism_kwargs.pop('q', 0.5)
        p = mechanism_kwargs.pop('p', percentage)
        targets = mechanism_kwargs.pop('targets', None)
        conditionals = mechanism_kwargs.pop('conditionals', None)
        correlation_threshold = mechanism_kwargs.pop('correlation_threshold', 0.5)
        if mechanism_kwargs:
            raise ValueError(f"Unexpected kwargs for mar: {sorted(mechanism_kwargs.keys())}")
        return remove_mar(
            X,
            q=q,
            p=p,
            targets=targets,
            conditionals=conditionals,
            correlation_threshold=correlation_threshold,
            random_state=random_state,
        )

    raise ValueError("`mechanism` must be one of: 'mcar', 'mnar', 'mar'.")


def prepare_missing_data(
    df,
    mechanism='mcar',
    percentage=0.10,
    random_state=42,
    continuous_transform='standardize',
    categorical_transform='encode',
    **mechanism_kwargs,
):
    """
    Prepare data, split train/test, and inject missingness into train/test features.

    This is a streamlined workflow for imputation experiments.
    """
    preprocessed_data = missing_prep_transform(
        df,
        continuous_transform=continuous_transform,
        categorical_transform=categorical_transform,
    )

    if preprocessed_data.empty:
        print("Preprocessed data is empty.")
        return preprocessed_data, None, None, None, None, None

    x_train, x_test, y_train, y_test = train_test(preprocessed_data, random_state=random_state)

    x_train_missing = apply_missingness(
        x_train,
        mechanism=mechanism,
        percentage=percentage,
        random_state=random_state,
        **mechanism_kwargs,
    )
    x_test_missing = apply_missingness(
        x_test,
        mechanism=mechanism,
        percentage=percentage,
        random_state=random_state,
        **mechanism_kwargs,
    )

    return x_train_missing, x_train, x_test, x_test_missing, y_train, y_test
