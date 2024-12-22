import pandas as pd

def inspect_csv(filepath):
    """
    Simple function to inspect CSV file structure
    """
    print("Reading file with comma delimiter...")
    try:
        df_comma = pd.read_csv(filepath)
        print("\nColumns with comma delimiter:")
        print(df_comma.columns.tolist())
        print("\nShape:", df_comma.shape)
    except Exception as e:
        print("Error reading with comma delimiter:", str(e))
    
    print("\nReading file with tab delimiter...")
    try:
        df_tab = pd.read_csv(filepath, sep='\t')
        print("\nColumns with tab delimiter:")
        print(df_tab.columns.tolist())
        print("\nShape:", df_tab.shape)
        
        # Print first few characters of the first column to see the structure
        print("\nFirst column name:")
        print(df_tab.columns[0])
        
        print("\nFirst few values of first column:")
        print(df_tab.iloc[0:2, 0])
    except Exception as e:
        print("Error reading with tab delimiter:", str(e))

if __name__ == "__main__":
    filepath = '/home/emirhan/anomaly-detection/data/RT_IOT2022.csv'
    inspect_csv(filepath)
