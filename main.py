from src.data_processing import load_data
from src.eda import basic_eda
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def main():
    # Load dataset
    file_path = r'C:\Users\ssahe\OneDrive\Documents\GitHub\ml-projects\final (1).csv'
    df = load_data(file_path)

    if df is None:
        return

    # Perform EDA
    basic_eda(df)

    # Preprocess data
    df = preprocess_data(df)

    # Split the data
    target_column = "price"
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Train the model (you can toggle between 'linear' and 'random_forest')
    model = train_model(X_train, y_train, model_type='random_forest')

    # Evaluate the model
    mae, mse, r2 = evaluate_model(model, X_test, y_test)

    # Display metrics
    print("\n✅ Model Evaluation:")
    print(f"📉 MAE: {mae}")
    print(f"📉 MSE: {mse}")
    print(f"📊 R2 Score: {r2}")

if __name__ == "__main__":
    main()