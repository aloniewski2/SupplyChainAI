import time
import random
import numpy as np
from datetime import datetime
import requests
import json

class DataGenerator:
    def __init__(self):
        self.base_demand = 100
        self.trend = 0.1
        self.seasonality = 20
        self.noise_level = 15
        
    def generate_demand(self, t):
        trend = self.trend * t
        seasonality = self.seasonality * np.sin(2 * np.pi * t / 365)
        noise = np.random.normal(0, self.noise_level)
        return max(0, self.base_demand + trend + seasonality + noise)
    
    def generate_metrics(self):
        current_time = datetime.now()
        t = current_time.timetuple().tm_yday
        
        demand = self.generate_demand(t)
        inventory = max(0, 500 + np.random.normal(0, 50))
        shipping_cost = 1000 + demand * 5 + np.random.normal(0, 100)
        
        return {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "demand": demand,
                "inventory": inventory,
                "shipping_cost": shipping_cost,
                "delivery_performance": {
                    "on_time_delivery_rate": min(100, 95 + np.random.normal(0, 2)),
                    "average_delivery_time": max(1, 3 + np.random.normal(0, 0.5)),
                    "delayed_shipments": max(0, int(np.random.normal(5, 2)))
                }
            }
        }

def run_generator():
    generator = DataGenerator()
    while True:
        try:
            data = generator.generate_metrics()
            response = requests.post('http://127.0.0.1:8000/metrics/update', 
                                  json=data)
            print(f"Data sent: {data}")
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_generator()