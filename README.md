# AI-Powered Log Analysis Dashboard

A comprehensive log analysis system that uses machine learning to detect anomalies, cluster similar log patterns, and visualize insights through an interactive dashboard.

## Technologies Used

- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms for clustering and anomaly detection
- **Dash & Plotly**: Interactive web dashboard
- **Matplotlib & Seaborn**: Data visualization
- **Faker**: Test data generation

## Project Purpose

This project provides an end-to-end solution for analyzing log files by:

1. Automatically detecting anomalies in log patterns
2. Clustering similar log messages
3. Visualizing log data through an interactive dashboard
4. Supporting multiple log formats (syslog, CSV, plain text)
5. Generating test log data for development and testing

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r Requirements.txt
```

### Running the Application

The application can be run with various options:

```bash
python app.py --log-file path/to/logfile.log --log-format syslog --port 8050
```

#### Command Line Arguments

- `--log-file`: Path to the log file to analyze
- `--log-format`: Format of the log file (syslog, csv, or None for plain text)
- `--port`: Port to run the dashboard on (default: 8050)
- `--generate-logs`: Generate sample log data
- `--num-entries`: Number of log entries to generate (default: 100000)

### Example Usage

Generate sample logs and analyze them:

```bash
python app.py --generate-logs --num-entries 10000
```

Analyze an existing log file:

```bash
python app.py --log-file logfiles.log
```

## Features

- **Log Preprocessing**: Normalizes log messages by removing variables like IP addresses and timestamps
- **Anomaly Detection**: Uses Isolation Forest and DBSCAN to identify unusual log patterns
- **Clustering**: Groups similar log messages to identify patterns
- **Interactive Dashboard**: Visualizes log data, anomalies, and clusters
- **Test Data Generation**: Creates realistic log data for testing and development