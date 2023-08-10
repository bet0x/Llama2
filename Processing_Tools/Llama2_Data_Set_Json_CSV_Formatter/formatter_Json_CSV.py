import json
import csv

json_data = r"D:/AI_CTS/Llama2/Processing_Tools/Data_Set_CSV_Json_Formatter/data_v1.json"
csv_data = r"D:/AI_CTS/Llama2/Processing_Tools/Llama2_Data_Set_Json_CSV_Formatter/train.csv"

# Read JSON data from file
with open(json_data, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create CSV file and write header
with open(csv_data, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['concept', 'description', 'Text'])

    # Write data to CSV
    for item in data:
        question = item['qas'][0]['question']
        answer = item['qas'][0]['answers'][0]['text']
        text = f"###Human:\n{question}\n\n###Assistant:\n{answer}"

        csv_writer.writerow([question, answer, text])

print("CSV conversion completed.")
