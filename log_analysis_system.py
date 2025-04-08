import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go

class LogAnalyzer:
    def __init__(self):
        self.log_data = None
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def load_logs(self, log_file, log_format=None):
        """Load logs from file with specified format"""
        if log_format == 'syslog':
            # Example syslog regex pattern
            pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\w+)\s+([^:]+):\s+(.*)'
            logs = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.match(pattern, line)
                    if match:
                        timestamp, host, program, message = match.groups()
                        logs.append({
                            'timestamp': timestamp,
                            'host': host,
                            'program': program,
                            'message': message
                        })
            
            self.log_data = pd.DataFrame(logs)
            
        elif log_format == 'csv':
            self.log_data = pd.read_csv(log_file)
        
        else:
            # Simple format - one log message per line
            with open(log_file, 'r') as f:
                logs = [line.strip() for line in f if line.strip()]
                self.log_data = pd.DataFrame({'message': logs})
        
        return self.log_data
    
    def preprocess_logs(self):
        """Preprocess log messages"""
        if self.log_data is None:
            raise ValueError("Log data not loaded. Call load_logs first.")
            
        # Convert timestamp to datetime if exists
        if 'timestamp' in self.log_data.columns:
            self.log_data['timestamp'] = pd.to_datetime(
                self.log_data['timestamp'], 
                errors='coerce'
            )
            
        # Extract severity level if exists in message
        severity_pattern = r'\b(ERROR|WARNING|INFO|DEBUG|CRITICAL|FATAL)\b'
        self.log_data['severity'] = self.log_data['message'].str.extract(
            severity_pattern, 
            expand=False
        )
        
        # Fill NaN values for severity
        self.log_data = self.log_data.assign(severity=self.log_data['severity'].fillna('UNKNOWN'))
        
        # Tokenize and normalize messages
        self.log_data['processed_message'] = self.log_data['message'].apply(
            lambda x: self._normalize_message(x)
        )
        
        return self.log_data
    
    def _normalize_message(self, message):
        """Normalize log message by removing variables"""
        # Remove IP addresses
        message = re.sub(r'\d+\.\d+\.\d+\.\d+', 'IP_ADDR', message)
        
        # Remove timestamps
        message = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIMESTAMP', message)
        
        # Remove file paths
        message = re.sub(r'\/[\w\/\.]+', 'FILE_PATH', message)
        
        # Remove numbers (keep some context)
        message = re.sub(r'\b\d+\b', 'NUM', message)
        
        return message
    
    def extract_features(self):
        """Extract features from log messages"""
        if 'processed_message' not in self.log_data.columns:
            self.preprocess_logs()
            
        # Use TF-IDF to vectorize log messages
        X = self.vectorizer.fit_transform(self.log_data['processed_message'])
        self.log_data['features'] = list(X.toarray())
        
        return X
    
    def cluster_logs(self):
        """Cluster similar log messages"""
        X = self.extract_features()
        
        # Apply clustering
        clusters = self.clustering_model.fit_predict(X)
        self.log_data['cluster'] = clusters
        
        # Label noise points (-1) as anomalies
        self.log_data['is_anomaly_cluster'] = self.log_data['cluster'] == -1
        
        return self.log_data
    
    def detect_anomalies(self):
        """Detect anomalies in log patterns"""
        if 'features' not in self.log_data.columns:
            self.extract_features()
            
        # Get features as numpy array
        X = np.array(self.log_data['features'].tolist())
        
        # Fit and predict anomalies
        anomalies = self.anomaly_detector.fit_predict(X)
        
        # -1 indicates anomaly, 1 indicates normal
        self.log_data['is_anomaly_isolation'] = anomalies == -1
        
        # Combine anomaly detection methods
        self.log_data['is_anomaly'] = (
            self.log_data['is_anomaly_cluster'] | 
            self.log_data['is_anomaly_isolation']
        )
        
        return self.log_data
    
    def analyze_logs(self, log_file, log_format=None):
        """Complete log analysis pipeline"""
        self.load_logs(log_file, log_format)
        self.preprocess_logs()
        self.cluster_logs()
        self.detect_anomalies()
        return self.log_data
    
    def get_summary(self):
        """Get summary statistics about logs"""
        if self.log_data is None:
            raise ValueError("Log data not loaded and processed.")
            
        summary = {
            'total_logs': len(self.log_data),
            'anomalies': sum(self.log_data['is_anomaly']),
            'anomaly_percentage': sum(self.log_data['is_anomaly']) / len(self.log_data) * 100,
            'clusters': len(set(self.log_data['cluster'])) - (1 if -1 in set(self.log_data['cluster']) else 0),
            'severity_counts': self.log_data['severity'].value_counts().to_dict()
        }
        
        # Add time-based statistics if timestamp exists
        if 'timestamp' in self.log_data.columns:
            summary['time_range'] = {
                'start': self.log_data['timestamp'].min(),
                'end': self.log_data['timestamp'].max()
            }
            
            # Add hourly distribution
            if not pd.isna(self.log_data['timestamp']).all():
                self.log_data['hour'] = self.log_data['timestamp'].dt.hour
                summary['hourly_distribution'] = self.log_data['hour'].value_counts().sort_index().to_dict()
        
        return summary
    
    def visualize_clusters(self):
        """Visualize log clusters"""
        if 'cluster' not in self.log_data.columns:
            raise ValueError("Logs not clustered. Call cluster_logs first.")
            
        from sklearn.manifold import TSNE
        
        # Get features
        X = np.array(self.log_data['features'].tolist())
        
        # Apply dimensionality reduction for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_2d = tsne.fit_transform(X)
        
        # Create visualization data
        viz_data = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': self.log_data['cluster'],
            'is_anomaly': self.log_data['is_anomaly']
        })
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        
        # Plot normal points
        normal_points = viz_data[~viz_data['is_anomaly']]
        sns.scatterplot(
            x='x', y='y', 
            hue='cluster', 
            data=normal_points,
            palette='viridis',
            alpha=0.7,
            s=50
        )
        
        # Plot anomalies
        anomaly_points = viz_data[viz_data['is_anomaly']]
        plt.scatter(
            anomaly_points['x'], 
            anomaly_points['y'],
            color='red',
            marker='x',
            s=100,
            label='Anomalies'
        )
        
        plt.title('Log Clusters with Anomalies')
        plt.legend()
        plt.savefig('log_clusters.png')
        plt.close()
        
        return 'log_clusters.png'

# Dashboard Implementation
def create_dashboard(log_analyzer):
    """Create a Dash dashboard for log visualization"""
    app = dash.Dash(__name__, title="Log Analysis Dashboard")
    
    # Get data for dashboard
    data = log_analyzer.log_data
    summary = log_analyzer.get_summary()
    
    # Create severity distribution pie chart
    severity_fig = px.pie(
        values=list(summary['severity_counts'].values()),
        names=list(summary['severity_counts'].keys()),
        title="Log Severity Distribution"
    )
    
    # Create anomaly timeline if timestamp exists
    if 'timestamp' in data.columns and not pd.isna(data['timestamp']).all():
        # Group by hour and count anomalies
        timeline_data = data.groupby([data['timestamp'].dt.date, data['timestamp'].dt.hour]).agg({
            'is_anomaly': 'sum',
            'message': 'count'
        }).reset_index()
        
        timeline_data['datetime'] = timeline_data.apply(
            lambda x: datetime.combine(x['timestamp'], datetime.min.time()) + 
                      pd.Timedelta(hours=x['hour']),
            axis=1
        )
        
        timeline_data['anomaly_rate'] = timeline_data['is_anomaly'] / timeline_data['message'] * 100
        
        anomaly_timeline = px.line(
            timeline_data,
            x='datetime',
            y=['anomaly_rate'],
            title="Anomaly Rate Over Time"
        )
    else:
        # Create empty figure if no timestamp
        anomaly_timeline = go.Figure()
        anomaly_timeline.update_layout(title="Anomaly Rate Over Time (No Time Data Available)")
    
    # Create cluster distribution bar chart
    cluster_counts = data['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    cluster_counts = cluster_counts[cluster_counts['cluster'] != -1]  # Remove noise cluster
    
    cluster_fig = px.bar(
        cluster_counts,
        x='cluster',
        y='count',
        title="Log Clusters Distribution"
    )
    
    # Create anomaly statistics card
    anomaly_card = html.Div([
        html.H4("Anomaly Statistics"),
        html.P(f"Total Logs: {summary['total_logs']}"),
        html.P(f"Anomalies Detected: {summary['anomalies']}"),
        html.P(f"Anomaly Rate: {summary['anomaly_percentage']:.2f}%")
    ], className="stats-card")
    
    # Create recent anomalies table
    recent_anomalies = data[data['is_anomaly']].sort_values(
        by='timestamp' if 'timestamp' in data.columns else 'message'
    ).tail(10)
    
    if 'timestamp' in recent_anomalies.columns:
        anomaly_table_data = recent_anomalies[['timestamp', 'message', 'severity']]
    else:
        anomaly_table_data = recent_anomalies[['message', 'severity']]
    
    # Main dashboard layout
    app.layout = html.Div([
        html.H1("AI-Powered Log Analysis Dashboard"),
        
        html.Div([
            html.Div([anomaly_card], className="four columns"),
            html.Div([
                dcc.Graph(figure=severity_fig)
            ], className="eight columns")
        ], className="row"),
        
        html.Div([
            dcc.Graph(figure=anomaly_timeline)
        ], className="row"),
        
        html.Div([
            dcc.Graph(figure=cluster_fig)
        ], className="row"),
        
        html.H3("Recent Anomalies"),
        html.Div([
            html.Table(
                [html.Tr([html.Th(col) for col in anomaly_table_data.columns])] +
                [html.Tr([
                    html.Td(anomaly_table_data.iloc[i][col]) for col in anomaly_table_data.columns
                ]) for i in range(min(len(anomaly_table_data), 10))]
            )
        ], className="row"),
        
    ], className="container")
    
    return app

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = LogAnalyzer()
    
    # Analyze logs
    # Replace with your log file path and format
    analyzer.analyze_logs("path/to/your/logfile.log", log_format="syslog")
    
    # Visualize clusters
    analyzer.visualize_clusters()
    
    # Create and run dashboard
    app = create_dashboard(analyzer)
    app.run_server(debug=True)