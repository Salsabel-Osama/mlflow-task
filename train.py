
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. Load the dataset
data = fetch_california_housing()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# 2. Split the dataset: 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# 3. Configure MLflow experiment
mlflow.set_experiment("California_Housing_Experiments")


# Hyperparameter configurations
experiments = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01},
]


# 4. Train and track each experiment
for i, config in enumerate(experiments, start=1):

    with mlflow.start_run(run_name=f"Run_{i}"):

        # Create the model
        model = GradientBoostingRegressor(
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict on validation data
        predictions = model.predict(X_val)

        # Calculate validation metrics
        rmse = mean_squared_error(
            y_val, predictions
        ) ** 0.5

        mae = mean_absolute_error(y_val, predictions)

        r2 = r2_score(y_val, predictions)

        # Log parameters
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("learning_rate", config["learning_rate"])

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Display results
        print(f"\nRun {i}")
        print(f"Max Depth: {config['max_depth']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")


print("\nAll experiments completed successfully!")