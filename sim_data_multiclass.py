import numpy as np
# import pandas as pd
def simulate_multiclass_data(n_samples=300,
                              n_features=2, 
                              n_classes=3, 
                              cluster_std=1.0, 
                              random_state=42):
    """
    Simulates multiclass data by generating clusters around random centers.
    """
    np.random.seed(random_state)
    
    # Randomly position unique centers for each class in space
    # (Uniform coordinates between -5.0 and 5.0)
    class_centers = np.random.uniform(-5.0, 5.0, size=(n_classes, n_features))
    
    X_list = []
    y_list = []
    
    # Calculate base samples per class (distribute remainders evenly)
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Determine specific sample count for current class to meet exact n_samples
        count = samples_per_class
        print(f"Class {class_idx}: {count} samples, Center: {class_centers[class_idx]}")
        
        # 2. Draw features from a Normal distribution centered at the class milestone
        features = np.random.normal(
            loc=class_centers[class_idx], 
            scale=cluster_std, 
            size=(count, n_features)
        )
        
        X_list.append(features)
        y_list.extend([class_idx] * count)
        
    # Stack array segments vertically 
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    
    return X, y

simulated_X, simulated_y = simulate_multiclass_data(n_samples=300, n_features=2, n_classes=3, cluster_std=1.0, random_state=42)

# convert y into one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
simulated_y_onehot = encoder.fit_transform(simulated_y.reshape(-1, 1))

np.save('./out_dir/simulated_multiclass_X.npy', simulated_X)
np.save('./out_dir/simulated_multiclass_y.npy', simulated_y_onehot)





# import matplotlib.pyplot as plt
# # Visualize the simulated multiclass data
# plt.figure(figsize=(8, 6))
# for class_idx in np.unique(simulated_y):
#     plt.scatter(simulated_X[simulated_y == class_idx, 0], 
#                 simulated_X[simulated_y == class_idx, 1], 
#                 label=f'Class {class_idx}', alpha=0.6)
# plt.title('Simulated Multiclass Data')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()