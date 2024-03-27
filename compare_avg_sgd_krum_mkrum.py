import numpy as np
import matplotlib.pyplot as plt
import time

class Client:
    def __init__(self, parameters, is_malicious=False):
        self.parameters = parameters
        self.is_malicious = is_malicious

    def get_parameters(self):
        return self.parameters

def create_composite_matrix(clients):
    shape = clients[0].get_parameters().shape
    composite_matrix = np.zeros((shape[0], shape[1], len(clients)))
    for idx, client in enumerate(clients):
        composite_matrix[:, :, idx] = client.get_parameters()
    return composite_matrix

def dynamic_sliding_window_aggregation(composite_matrix, window_size=4, epsilon=1e-5):
    """
    Performs dynamic sliding window aggregation on composite matrices with weighted mean based on Euclidean distance from the median,
    applied separately for each parameter across clients. Then, computes the average of these weighted means to get the final aggregated value for each parameter.
    """
    rows, cols, clients = composite_matrix.shape
    final_aggregated_matrix = np.zeros((rows, cols))

    # Process each parameter separately
    for i in range(rows):
        for j in range(cols):
            window_means = []  # List to hold the weighted mean of each window for the current parameter position

            # Process clients in windows for the current parameter
            for window_index in range(0, clients, window_size):
                # Select clients for the current window
                window_clients = composite_matrix[i, j, window_index:window_index + window_size]

                # Calculate median of the window
                if len(window_clients) > 0:  # Check to ensure window is not empty
                    median_window = np.median(window_clients)
                    # Calculate Euclidean distance from the median
                    distances = np.abs(window_clients - median_window)
                    # Dynamically assign weights based on the distance
                    weights = 1 / (epsilon + distances**2)

                    # Calculate weighted mean for the current window and add to the list
                    weighted_mean = np.sum(window_clients * weights) / np.sum(weights)
                    window_means.append(weighted_mean)

            # Calculate the average of all window means for the current parameter position
            if len(window_means) > 0:
                final_aggregated_matrix[i, j] = np.mean(window_means)
            else:
                final_aggregated_matrix[i, j] = 0

    return final_aggregated_matrix

def detect_malicious_clients(aggregated_matrix, clients):
    deviations = []
    for idx, client in enumerate(clients):
        client_matrix = client.get_parameters()
        deviation = np.mean(np.abs(client_matrix - aggregated_matrix))
        deviations.append(deviation)
    threshold = np.mean(deviations) + np.std(deviations)
    malicious_indices = [i for i, deviation in enumerate(deviations) if deviation > threshold]
    return threshold, malicious_indices

def federated_learning_aggregation(clients, window_size=4):
    composite_matrix = create_composite_matrix(clients)
    aggregated_matrix = dynamic_sliding_window_aggregation(composite_matrix, window_size)
    threshold, malicious_indices = detect_malicious_clients(aggregated_matrix, clients)
    return aggregated_matrix, threshold, malicious_indices

def fedavg_aggregate(clients):
    total_parameters = None
    for client in clients:
        parameters = client.get_parameters()
        if total_parameters is None:
            total_parameters = parameters.copy()
        else:
            total_parameters += parameters
    return total_parameters / len(clients)

def fedsgd_aggregate(clients):
    total_updates = None
    for client in clients:
        updates = client.get_parameters()
        if total_updates is None:
            total_updates = updates.copy()
        else:
            total_updates += updates
    return total_updates / len(clients)

def krum_aggregate(clients, num_discard=1):
    num_clients = len(clients)
    num_params = clients[0].get_parameters().size
    all_params = np.array([client.get_parameters().flatten() for client in clients])

    # Compute distances
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            distances[i, j] = np.linalg.norm(all_params[i] - all_params[j])
            distances[j, i] = distances[i, j]

    # Find indices to exclude
    exclude_indices = np.argsort(np.sum(distances, axis=1))[:num_discard]

    # Aggregate parameters
    selected_clients = [client for idx, client in enumerate(clients) if idx not in exclude_indices]
    aggregated_params = fedavg_aggregate(selected_clients)

    return aggregated_params.reshape((int(np.sqrt(num_params)), int(np.sqrt(num_params))))

def multi_krum_aggregate(clients, num_discard=2):
    num_clients = len(clients)
    num_params = clients[0].get_parameters().size
    all_params = np.array([client.get_parameters().flatten() for client in clients])

    # Compute distances
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            distances[i, j] = np.linalg.norm(all_params[i] - all_params[j])
            distances[j, i] = distances[i, j]

    # Find indices to exclude
    discard_indices = np.argsort(np.sum(distances, axis=1))[:num_discard]
    remaining_indices = [idx for idx in range(num_clients) if idx not in discard_indices]

    # Aggregate parameters
    remaining_params = all_params[remaining_indices]
    aggregated_params = np.mean(remaining_params, axis=0)

    return aggregated_params.reshape((int(np.sqrt(num_params)), int(np.sqrt(num_params))))


def evaluate_model_performance(aggregated_matrix, ideal_matrix):
    # Calculate the Euclidean distance to the ideal matrix
    performance_score = np.linalg.norm(aggregated_matrix - ideal_matrix)
    return performance_score

# Calculate the average value of the parameter matrix for normal clients
def calculate_ideal_matrix(normal_clients):
    sum_matrix = None
    for client in normal_clients:
        params = client.get_parameters()
        if sum_matrix is None:
            sum_matrix = params.copy()
        else:
            sum_matrix += params
    return sum_matrix / len(normal_clients)


def evaluate_detection(actual_malicious_indices, detected_malicious_indices):
    true_positives = len(set(actual_malicious_indices) & set(detected_malicious_indices))
    precision = true_positives / len(detected_malicious_indices) if detected_malicious_indices else 0
    recall = true_positives / len(actual_malicious_indices) if actual_malicious_indices else 0
    return precision, recall


def compare_algorithms(clients):
    # Running time of the sliding window algorithm
    start_time = time.time()
    sliding_window_aggregated_matrix, sliding_window_threshold, sliding_window_detected_malicious_indices = federated_learning_aggregation(
        clients)
    sliding_window_time = time.time() - start_time

    # Runtime of the FedAvg algorithm
    start_time = time.time()
    fedavg_aggregated_matrix = fedavg_aggregate(clients)
    fedavg_time = time.time() - start_time

    # Runtime of the FedSGD algorithm
    start_time = time.time()
    fedsgd_aggregated_matrix = fedsgd_aggregate(clients)
    fedsgd_time = time.time() - start_time

    # Runtime of the Krum algorithm
    start_time = time.time()
    krum_aggregated_matrix = krum_aggregate(clients)
    krum_time = time.time() - start_time

    # Runtime of the Multi-Krum algorithm
    start_time = time.time()
    multi_krum_aggregated_matrix = multi_krum_aggregate(clients)
    multi_krum_time = time.time() - start_time

    return sliding_window_time, fedavg_time, fedsgd_time, krum_time, multi_krum_time


# Initialise the client
np.random.seed(0)

normal_feature_means = [0.5, 1.7, 7, 4]
normal_clients = []
for _ in range(584):
    params = np.random.rand(4, 4)
    for feature_index, mean in enumerate(normal_feature_means):
        params[feature_index, :] = np.random.normal(loc=mean, scale=0.1, size=params.shape[1])
    normal_clients.append(Client(params))

# Initialise malicious clients, set extreme values for certain parameters
malicious_clients = []
extreme_values = [100, -100, 300, -200]
for _ in range(16):
    params = np.random.rand(4, 4) * 0.5  # Use the basic parameter matrix as a starting point
    for feature_index, extreme_value in enumerate(extreme_values):
        # Set extreme values for specific parameter locations for each malicious client
        params[feature_index, 0] = extreme_value
    malicious_clients.append(Client(params, is_malicious=True))

# Combine all clients and randomly disrupt
clients = normal_clients + malicious_clients
np.random.shuffle(clients)
# Call functions to compare algorithms
sliding_window_time, fedavg_time, fedsgd_time, krum_time, multi_krum_time = compare_algorithms(clients)

ideal_matrix = calculate_ideal_matrix(normal_clients)

# Getting the real malicious client indices
actual_malicious_indices = [i for i, client in enumerate(clients) if client.is_malicious]

# Run the federated learning aggregation process with sliding window and detect malicious clients
aggregated_matrix_sw, threshold_sw, detected_malicious_indices_sw = federated_learning_aggregation(clients)
precision_sw, recall_sw = evaluate_detection(actual_malicious_indices, detected_malicious_indices_sw)
model_performance_sw = evaluate_model_performance(aggregated_matrix_sw, ideal_matrix)

# Run the FedAvg aggregation process
aggregated_matrix_fa = fedavg_aggregate(clients)
model_performance_fa = evaluate_model_performance(aggregated_matrix_fa, ideal_matrix)
# Assuming no malicious client detection capability for FedAvg
precision_fa, recall_fa = (0, 0)

# Run the FedSGD aggregation process
aggregated_matrix_fs = fedsgd_aggregate(clients)
model_performance_fs = evaluate_model_performance(aggregated_matrix_fs, ideal_matrix)
# Assuming no malicious client detection capability for FedSGD
precision_fs, recall_fs = (0, 0)

# Run the Krum aggregation process
aggregated_matrix_krum = krum_aggregate(clients)
model_performance_krum = evaluate_model_performance(aggregated_matrix_krum, ideal_matrix)
# Evaluate malicious client detection
_, malicious_indices_krum = detect_malicious_clients(aggregated_matrix_krum, clients)
precision_krum, recall_krum = evaluate_detection(actual_malicious_indices, malicious_indices_krum)

# Run the Multi-Krum aggregation process
aggregated_matrix_multi_krum = multi_krum_aggregate(clients)
model_performance_multi_krum = evaluate_model_performance(aggregated_matrix_multi_krum, ideal_matrix)
# Evaluate malicious client detection
_, malicious_indices_multi_krum = detect_malicious_clients(aggregated_matrix_multi_krum, clients)
precision_multi_krum, recall_multi_krum = evaluate_detection(actual_malicious_indices, malicious_indices_multi_krum)


# Printing the comparison results
print("Comparison of Algorithms:")
print(f"Sliding Window - Model Performance: {model_performance_sw:.4f}, Precision: {precision_sw * 100:.2f}%, Recall: {recall_sw * 100:.2f}%, Sliding Window - Time Efficiency: {sliding_window_time:.4f} seconds ")
print(f"FedAvg - Model Performance: {model_performance_fa:.4f}, Precision: {precision_fa * 100:.2f}%, Recall: {recall_fa * 100:.2f}%, FedAvg - Time Efficiency: {fedavg_time:.4f} seconds")
print(f"FedSGD - Model Performance: {model_performance_fs:.4f}, Precision: {precision_fs * 100:.2f}%, Recall: {recall_fs * 100:.2f}%, FedSGD - Time Efficiency: {fedsgd_time:.4f} seconds")
print(f"Krum - Model Performance: {model_performance_krum:.4f}, Precision: {precision_krum * 100:.2f}%, Recall: {recall_krum * 100:.2f}%, Krum - Time Efficiency: {krum_time:.4f} seconds")
print(f"Multi-Krum - Model Performance: {model_performance_multi_krum:.4f}, Precision: {precision_multi_krum * 100:.2f}%, Recall: {recall_multi_krum * 100:.2f}%, Multi-Krum - Time Efficiency: {multi_krum_time:.4f} seconds")

# Visualizing the results for a more intuitive comparison
algorithms = ['Sliding Window', 'FedAvg', 'FedSGD', 'Krum', 'Multi-Krum']

performances = [model_performance_sw, model_performance_fa, model_performance_fs, model_performance_krum, model_performance_multi_krum]
recalls = [recall_sw, recall_fa, recall_fs, recall_krum, recall_multi_krum]
precisions = [precision_sw, precision_fa, precision_fs, precision_krum, precision_multi_krum]
time_efficiencies = [sliding_window_time, fedavg_time, fedsgd_time, krum_time, multi_krum_time]

# Normalizing the performance for better visual comparison
max_performance = max(performances)
normalized_performances = [p / max_performance for p in performances]

# Create charts
fig, ax1 = plt.subplots(figsize=(14, 8))

# Setting the location
ind = np.arange(len(algorithms))  # x-axis position
width = 0.2  # Width of the bar

performance_bars = ax1.bar(ind - width*1.5, normalized_performances, width, label='Normalized Model Performance', color='tab:blue')
precision_bars = ax1.bar(ind - width/2, precisions, width, label='Precision', color='tab:green')
recall_bars = ax1.bar(ind + width/2, recalls, width, label='Recall', color='tab:red')
time_efficiency_bars = ax1.bar(ind + width*1.5, time_efficiencies, width, label='Time Efficiency', color='tab:orange')

ax1.set_xlabel('Algorithms', fontsize=14)
ax1.set_ylabel('Scores & Inversed Time Efficiency', fontsize=14)
ax1.set_xticks(ind)
ax1.set_xticklabels(algorithms, fontsize=12)
ax1.legend()
plt.title('Comparison of Algorithm Performance, Precision, Recall, and Time Efficiency', fontsize=16)
plt.tight_layout()
plt.show()