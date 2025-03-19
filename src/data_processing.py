import pandas as pd

def load_data(file_path):
    """
    Loads dataset from a CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("❌ The file is empty.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None