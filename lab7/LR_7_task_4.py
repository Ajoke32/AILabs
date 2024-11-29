import numpy as np
import yfinance as yf
from sklearn import covariance, cluster

company_symbols_map = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "CVX": "Chevron Corporation",
    "KO": "Coca-Cola Company",
    "PFE": "Pfizer Inc.",
    "PEP": "PepsiCo Inc."
}

symbols, names = np.array(list(company_symbols_map.items())).T

# data loading parameters
start_date = "2020-09-01"
end_date = "2023-10-01"

quotes = []

# data loading for each symbol
for symbol in symbols:
    print(f"Loading data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"Data for {symbol} has been loaded")
    quotes.append(data)

# Removal of quotes corresponding to the opening and closing of the exchange
opening_quotes = [data['Open'].values for data in quotes]
closing_quotes = [data['Close'].values for data in quotes]

# Make sure that all arrays have the same length
min_length = min(map(len, opening_quotes))
opening_quotes = np.array([x[:min_length] for x in opening_quotes], dtype=np.float64)
closing_quotes = np.array([x[:min_length] for x in closing_quotes], dtype=np.float64)

# Calculating the difference between quotes and bringing them to the correct form
quotes_diff = np.array([closing - opening for closing, opening in zip(closing_quotes, opening_quotes)])

# transpose for the model and remove the extra dimension, if any
X = quotes_diff.T.squeeze()

print(f"Dimension of X: {X.shape}")  # Checking the dimensionality of X

# Standardize data
std_dev = X.std(axis=0)
std_dev[std_dev == 0] = 1
X /= std_dev

# Replace NaN with the average of the columns
nan_mask = np.isnan(X)
col_means = np.nanmean(X, axis=0)
X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

edge_model = covariance.GraphicalLassoCV()
edge_model.fit(X)

# Clustering
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Results:
for i in range(num_labels + 1):
    print(f"Cluster {i + 1} => {', '.join(names[labels == i])}")
