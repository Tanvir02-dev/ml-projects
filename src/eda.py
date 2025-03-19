def basic_eda(df):
    """
    Performs basic exploratory data analysis on the dataset.
    """
    print("\n📊 First 5 rows:")
    print(df.head())

    print("\n📏 Dataset Shape:", df.shape)

    print("\n📃 Data Info:")
    df.info()

    print("\n📈 Summary Statistics:")
    print(df.describe())

    print("\n🔍 Missing Values:")
    print(df.isnull().sum())