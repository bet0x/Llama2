
import csv
import os

def write_cells_to_text_files(csv_file_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the CSV file and extract the "Answer" column
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader, start=1):
                cell_content = row.get('Answer', '')
                # Create a text file with the format: "output_folder/answer_<index>.txt"
                file_path = os.path.join(output_folder, f"answer_{idx}.txt")
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        # Write the cell content to the text file
                        file.write(cell_content)
                except Exception as e:
                    print(f"Error writing to file {file_path}: {e}")
    except FileNotFoundError:
        print("CSV file not found. Please provide the correct path.")
        return

if __name__ == "__main__":
    # Provide the path to your CSV file and the output folder
    csv_file_path = r"Simple_Transformer/Question_Answer_v5/Data_Set/wiki.csv"
    output_folder = r"D:/AI_CTS/NLP/Simple_Transformer/Question_Answer_v5/Data_Set/Data_Set_Output_Text/Output"
    write_cells_to_text_files(csv_file_path, output_folder)
