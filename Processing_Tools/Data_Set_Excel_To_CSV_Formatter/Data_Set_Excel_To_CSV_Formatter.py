import pandas as pd

def excel_to_csv(input_file, output_file):
    try:
        # Read the Excel file
        df = pd.read_excel(input_file)
        
        # Extract "Question" and "Answer" columns
        df_extracted = df[["Question", "Answer"]]
        
        # Save the extracted data to a CSV file
        df_extracted.to_csv(output_file, index=False)
        
        print(f"Successfully converted '{input_file}' to '{output_file}'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace 'input.xlsx' with the path of your input Excel file
    input_file_path = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Data_Set_Excel_To_CSV_Formatter/V2/Hotline_Wiki_v2.xlsx"
   
    # Replace 'output.csv' with the desired name of your output CSV file
    output_file_path = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Data_Set_Excel_To_CSV_Formatter/V2/Wiki_v2.csv"
    
    excel_to_csv(input_file_path, output_file_path)
