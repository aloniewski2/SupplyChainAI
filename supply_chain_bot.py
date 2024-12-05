import pandas as pd
import numpy as np
from models import SupplyChainPredictor
import matplotlib.pyplot as plt
import seaborn as sns

def run_ml_demo():
    print("Generating sample data...")
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-02-29', freq='D')
    np.random.seed(42)
    
    # Create realistic patterns
    trend = np.linspace(0, 50, len(dates))
    seasonality = 20 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    weekly = 10 * np.sin(np.linspace(0, 52*2*np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    
    data = pd.DataFrame({
        'date': dates,
        'demand': 100 + trend + seasonality + weekly + noise,
        'price': np.random.uniform(10, 20, len(dates)),
        'inventory': np.random.normal(500, 50, len(dates)),
        'shipping_cost': np.random.normal(1000, 200, len(dates))
    })
    
    # Split data into train and test
    print("\nSplitting data into train and test sets...")
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize and train predictor
    print("\nInitializing models...")
    predictor = SupplyChainPredictor()
    
    print("\nTraining models...")
    predictor.train(train_data)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(test_data)
    
    # Ensure all predictions have the same length as actual values
    actual = test_data['demand'].values
    pred_length = len(actual)
    
    # Plot results
    print("\nGenerating visualizations...")
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 10))
    
    # Plot actual vs predicted
    plt.subplot(2, 1, 1)
    plt.plot(test_data['date'], actual, label='Actual', color='black', linewidth=2)
    
    for model_name, preds in predictions['model_predictions'].items():
        # Truncate predictions to match actual length
        preds = preds[:pred_length]
        plt.plot(test_data['date'], preds, label=model_name, alpha=0.5)
    
    # Ensure ensemble prediction matches length
    ensemble_pred = predictions['ensemble_prediction'][:pred_length]
    plt.plot(test_data['date'], ensemble_pred, 
             label='Ensemble', linewidth=2, color='red', linestyle='--')
    
    plt.title('Demand Predictions by Different Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Calculate and plot performance metrics
    plt.subplot(2, 1, 2)
    
    performance = {}
    for model_name, preds in predictions['model_predictions'].items():
        preds = preds[:pred_length]  # Ensure same length as actual
        mse = np.mean((preds - actual) ** 2)
        mae = np.mean(np.abs(preds - actual))
        performance[model_name] = {
            'mse': mse,
            'mae': mae,
            'accuracy': 1 - mae / np.mean(actual)
        }
    
    # Add ensemble performance
    ensemble_pred = predictions['ensemble_prediction'][:pred_length]
    mse = np.mean((ensemble_pred - actual) ** 2)
    mae = np.mean(np.abs(ensemble_pred - actual))
    performance['ensemble'] = {
        'mse': mse,
        'mae': mae,
        'accuracy': 1 - mae / np.mean(actual)
    }
    
    # Plot performance metrics
    models = list(performance.keys())
    accuracies = [perf['accuracy'] for perf in performance.values()]
    
    bars = plt.bar(models, accuracies)
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', bbox_inches='tight')
    print("\nResults saved as 'model_comparison.png'")
    
    # Print detailed performance metrics
    print("\nPerformance Metrics:")
    for model, metrics in performance.items():
        print(f"\n{model.upper()}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"MSE: {metrics['mse']:.3f}")
        print(f"MAE: {metrics['mae']:.3f}")

if __name__ == "__main__":
    run_ml_demo()