import csv
import json

def convert_csv_to_json(csv_path, json_path):
    data = []
    id_counter = 1  # Initialize the ID counter

    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            context = row["Answer"]
            qas = [
                {
                    "id": f"{id_counter:05}",  # Format the ID to be 5-digit numerical
                    "is_impossible": False,  # You can set this value based on your requirements
                    "question": row["Question"],
                    "answers": [
                        {
                            "text": row["Answer"],
                            "answer_start": 0,  # You can set this value based on your requirements
                        }
                    ],
                }
            ]
            data.append({"context": context, "qas": qas})
            id_counter += 1  # Increment the ID counter for the next question

    # Write to JSON file
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2)

# Replace these paths with your input CSV file and desired output JSON file
csv_file_path = r"C:\Users\Lukas\Desktop\My_Projects\NLP\Google_Tapas_Base\wiki.csv"
json_file_path = r"C:\Users\Lukas\Desktop\My_Projects\NLP\Google_Tapas_Base\data_v1.json"

convert_csv_to_json(csv_file_path, json_file_path)
