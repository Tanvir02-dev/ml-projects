from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train, model_type='linear'):
    """
    Trains a model (Linear Regression or Random Forest) on the training data.
    Returns the trained model.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("❌ Unsupported model type. Use 'linear' or 'random_forest'.")

    model.fit(X_train, y_train)
    return model