import json
import csv

json_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Llama2_Data_Set_Json_CSVChat_Formatter/data_v2.json"
csv_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Llama2_Data_Set_Json_CSVChat_Formatter/train.csv"

# Read JSON data from file
with open(json_data, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create CSV file and write header
with open(csv_data, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['text'])

    # Write data to CSV
    for item in data:
        question = item['qas'][0]['question']
        answers = item['qas'][0]['answers'][0]['text']
        text = f"<s>[INST]{question}[/INST]{answers}</s>"

        csv_writer.writerow([text])

print("CSV conversion completed.")
