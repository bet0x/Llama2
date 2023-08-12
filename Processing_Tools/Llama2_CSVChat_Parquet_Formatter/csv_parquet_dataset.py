
import pandas as pd

csv_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Llama2_Data_Set_Json_CSVChat_Formatter/train.csv"
parquet_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Llama2_CSVChat_Parquet_Formatter/train-00000-of-00001-e005b3c07c773eca.parquet"

# Read CSV file
csv_file_path = csv_data
data_frame = pd.read_csv(csv_file_path)

# Specify Parquet file path
parquet_file_path = parquet_data

# Convert to Parquet format
data_frame.to_parquet(parquet_file_path, index=False)

print("CSV to Parquet conversion completed.")
