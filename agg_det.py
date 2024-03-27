import numpy as np
import matplotlib.pyplot as plt

class Client:
    def __init__(self, parameters, is_malicious=False):
        """
            The Client class represents a client in federated learning,
                with an added feature to identify malicious clients.
            - parameters: numpy array, client parameter matrix.
            - is_malicious: boolean, flag to identify if the client is malicious.
        """
        self.parameters = parameters
        self.is_malicious = is_malicious

    def get_parameters(self):
        """
            Retrieve the client's parameter matrix.
            Returns:
            - parameters: numpy array, the client-side parameter matrix.
        """
        return self.parameters

# Create a composite matrix of all clients
def create_composite_matrix(clients):
    """
        Creates a composite matrix from all client parameter matrices.
        - clients: list of Client objects.
        Returns:
        - composite_matrix: numpy array, 3D matrix where each layer represents a client's parameter matrix.
    """
    shape = clients[0].get_parameters().shape
    composite_matrix = np.zeros((shape[0], shape[1], len(clients)))
    for idx, client in enumerate(clients):
        composite_matrix[:, :, idx] = client.get_parameters()
    return composite_matrix

# Dynamic sliding window aggregation function
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

# Detecting malicious clients
def detect_malicious_clients(aggregated_matrix, clients):
    """
        Detects malicious clients based on their deviation from the aggregated matrix.
        - aggregated_matrix: numpy array, the aggregated matrix of client parameters.
        - clients: list of Client objects.
        Returns:
        - threshold: float, calculated threshold to detect malicious clients.
        - malicious_indices: list, indices of detected malicious clients.
    """
    deviations = []
    for idx, client in enumerate(clients):
        client_matrix = client.get_parameters()
        deviation = np.mean(np.abs(client_matrix - aggregated_matrix))
        deviations.append(deviation)
    threshold = np.mean(deviations) + np.std(deviations)
    malicious_indices = [i for i, deviation in enumerate(deviations) if deviation > threshold]
    return threshold, malicious_indices

# Aggregate and detect malicious clients
def federated_learning_aggregation(clients, window_size=4):
    """
        Main aggregation function for federated learning that also detects malicious clients.
        - clients: list of Client objects.
        - window_size: tuple, dimensions of the sliding window for aggregation.
        Returns:
        - aggregated_matrix: numpy array, the final aggregated parameter matrix.
        - threshold: float, threshold used to detect malicious clients.
        - malicious_indices: list, indices of detected malicious clients.
    """
    composite_matrix = create_composite_matrix(clients)
    aggregated_matrix = dynamic_sliding_window_aggregation(composite_matrix, window_size)
    threshold, malicious_indices = detect_malicious_clients(aggregated_matrix, clients)
    return aggregated_matrix, threshold, malicious_indices

# 评估检测结果
def evaluate_detection(actual_malicious_indices, detected_malicious_indices):
    """
        Evaluates the detection of malicious clients by calculating precision and recall.
        - actual_malicious_indices: list, indices of actual malicious clients.
        - detected_malicious_indices: list, indices of detected malicious clients.
        Returns:
        - precision: float, precision of the detection.
        - recall: float, recall of the detection.
    """
    true_positives = len(set(actual_malicious_indices) & set(detected_malicious_indices))
    precision = true_positives / len(detected_malicious_indices) if detected_malicious_indices else 0
    recall = true_positives / len(actual_malicious_indices) if actual_malicious_indices else 0
    return precision, recall

def plot_malicious_clients(deviations, malicious_indices):
    plt.figure(figsize=(12, 6))
    bar_colors = ['blue' if i not in malicious_indices else 'red' for i in range(len(deviations))]
    plt.bar(range(len(deviations)), deviations, color=bar_colors)
    plt.xlabel('Client Index')
    plt.ylabel('Deviation from Aggregated Matrix')
    plt.title('Client Deviation and Detected Malicious Clients')
    plt.show()


def plot_client_weights(deviations, malicious_indices):
    # Assuming that the inverse of the deviation is used as a proxy for the weights
    weights = 1 / (np.array(deviations) + 1e-5)
    normalized_weights = weights / np.max(weights)  # normalized weight

    plt.figure(figsize=(12, 6))
    # Plotting the weights of all clients
    plt.bar(range(len(normalized_weights)), normalized_weights, color='blue', label='Normal Clients')

    # Individual highlighting of malicious client weights
    malicious_weights = normalized_weights[malicious_indices]
    plt.scatter(malicious_indices, malicious_weights, color='red', s=100, label='Malicious Clients', zorder=5)

    plt.xlabel('Client Index')
    plt.ylabel('Normalized Weight')
    plt.title('Weight Assigned to Each Client')
    plt.legend()
    plt.show()

# Instantiating the client
np.random.seed(0)

normal_feature_means = [0.5, 1.7, 55, 4]
normal_clients = []
for _ in range(584):
    params = np.random.rand(4, 4)
    for feature_index, mean in enumerate(normal_feature_means):
        params[feature_index, :] = np.random.normal(loc=mean, scale=0.1, size=params.shape[1])
    normal_clients.append(Client(params))

# Initialising a malicious client, setting extreme values for certain parameters
malicious_clients = []
extreme_values = [100, -100, 300, -200]  # Examples of extreme values
for _ in range(16):  # Assuming there are 16 malicious clients
    params = np.random.rand(4, 4) * 0.5  # Use the basic parameter matrix as a starting point
    for feature_index, extreme_value in enumerate(extreme_values):
        # Setting extreme values for specific parameter locations for each malicious client
        params[feature_index, 0] = extreme_value
    malicious_clients.append(Client(params, is_malicious=True))

clients = normal_clients + malicious_clients
np.random.shuffle(clients)

# Getting the real malicious client index
actual_malicious_indices = [i for i, client in enumerate(clients) if client.is_malicious]

# Run the federated learning aggregation process and obtain the aggregation matrix and the index of detected malicious clients
aggregated_matrix, threshold, detected_malicious_indices = federated_learning_aggregation(clients)

# Calculation bias
deviations = [np.mean(np.abs(client.get_parameters() - aggregated_matrix)) for client in clients]

# Calculate weights, which are calculated here for visualization purposes only
weights = 1 / (np.array(deviations) + 1e-5)
normalized_weights = weights / np.max(weights)

# Calculate precision and recall
precision, recall = evaluate_detection(actual_malicious_indices, detected_malicious_indices)

# printout
final_aggregated_matrix = aggregated_matrix
print("Final Aggregated Matrix:\n", final_aggregated_matrix)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

# Print indices of the actual and detected malicious clients
print("Actual Malicious Client Indices:", actual_malicious_indices)
print("Detected Malicious Client Indices:", detected_malicious_indices)

# Print weights assigned to malicious clients and the first 10 normal clients
malicious_weights = normalized_weights[actual_malicious_indices]
normal_weights = normalized_weights[np.isin(range(len(clients)), actual_malicious_indices, invert=True)]

print("Weights Assigned to Malicious Clients:", malicious_weights)
print("Weights Assigned to First 10 Normal Clients:", normal_weights[:10])

print("Malicious Client Information：")
for i in actual_malicious_indices:
    print(f"Client {i}: actual deviation = {deviations[i]}, threshold = {threshold}")

# Print information about normal clients (assuming normal clients are those not in actual_malicious_indices)
print("\nNormal Client Information（First 10）：")
normal_client_indices = [i for i in range(len(clients)) if i not in actual_malicious_indices][:10]  # Only the first 10 normal clients are selected to simplify the output
for i in normal_client_indices:
    print(f"Client {i}: actual deviation = {deviations[i]}, threshold = {threshold}")
# Calculate deviation values for all clients
deviations = [np.mean(np.abs(client.get_parameters() - aggregated_matrix)) for client in clients]

# Plotting deviations
plot_malicious_clients(deviations, detected_malicious_indices)

# Plotting weights
plot_client_weights(deviations, detected_malicious_indices)
