from log_analysis_system import LogAnalyzer, create_dashboard
from TestFileGenerator import LogGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Log Analysis Dashboard')
    parser.add_argument('--log-file', help='Path to log file')
    parser.add_argument('--log-format', default=None, 
                        choices=['syslog', 'csv', None], 
                        help='Format of log file')
    parser.add_argument('--port', type=int, default=8050, 
                        help='Port to run dashboard on')
    parser.add_argument('--generate-logs', action='store_true',
                        help='Generate sample log data')
    parser.add_argument('--num-entries', type=int, default=100000,
                        help='Number of log entries to generate')
    
    args = parser.parse_args()
    
    # Generate logs if requested
    if args.generate_logs:
        output_file = args.log_file or "logfiles.log"
        print(f"Generating {args.num_entries} log entries...")
        generator = LogGenerator()
        generator.generate_log_file(output_file, args.num_entries)
        print(f"Log generation complete! Output saved to {output_file}")
        args.log_file = output_file
    
    if not args.log_file:
        parser.error("Either --log-file or --generate-logs must be specified")
    
    # Initialize analyzer
    analyzer = LogAnalyzer()
    
    # Analyze logs
    print(f"Analyzing logs from {args.log_file}...")
    analyzer.analyze_logs(args.log_file, log_format=args.log_format)
    
    # Get summary
    summary = analyzer.get_summary()
    print("\nLog Analysis Summary:")
    print(f"Total logs: {summary['total_logs']}")
    print(f"Anomalies detected: {summary['anomalies']} ({summary['anomaly_percentage']:.2f}%)")
    print(f"Number of clusters: {summary['clusters']}")
    
    # Visualize clusters
    print("\nGenerating cluster visualization...")
    viz_file = analyzer.visualize_clusters()
    print(f"Visualization saved to {viz_file}")
    
    # Create and run dashboard
    print(f"\nStarting dashboard on port {args.port}...")
    app = create_dashboard(analyzer)
    app.run(debug=False, port=args.port)

if __name__ == "__main__":
    main()