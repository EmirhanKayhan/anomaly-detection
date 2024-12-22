import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('/home/emirhan/anomaly-detection/data/X_train.csv')
    y_train = pd.read_csv('/home/emirhan/anomaly-detection/data/y_train.csv')
    return pd.concat([X_train, y_train], axis=1)

def visualize_data(data):
    """Create comprehensive visualizations"""
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution of Attack Types
    plt.subplot(2, 2, 1)
    data['Attack_type'].value_counts().plot(kind='bar')
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 2. Correlation Heatmap
    plt.subplot(2, 2, 2)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    
    # 3. Scatter Plot of Two Important Features
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=data, x='flow_duration', y='flow_pkts_per_sec', hue='Attack_type')
    plt.title('Flow Duration vs Packets per Second')
    
    # 4. Box Plot of Numerical Features by Attack Type
    plt.subplot(2, 2, 4)
    data.boxplot(column='fwd_pkts_tot', by='Attack_type')
    plt.title('Forward Packets Total by Attack Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/emirhan/anomaly-detection/data/data_visualization.png')
    plt.close()

def main():
    data = load_preprocessed_data()
    visualize_data(data)
    print("Data visualization completed. Check data_visualization.png")

if __name__ == "__main__":
    main()
