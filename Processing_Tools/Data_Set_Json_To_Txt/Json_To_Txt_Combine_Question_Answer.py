import json
import os

json_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Data_Set_Json_To_Txt/data_v2.json"
txt_data = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Data_Set_Json_To_Txt/data_combine"

def write_to_text_file(output_filename, context, qa_combined):
    with open(output_filename, 'w') as f:
        #f.write(f"Context: {context}\n")
        #f.write(f"Question and Answer: {qa_combined}\n\n")
        f.write(f"{qa_combined}\n\n")

def main(input_json_file, output_folder):
    with open(input_json_file, 'r') as f:
        json_list = json.load(f)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, item in enumerate(json_list):
        context = item['context']
        qas = item['qas']
        for qa in qas:
            question = qa['question']
            answer = qa['answers'][0]['text']
            #qa_combined = f"Q: {question}\nA: {answer}"
            qa_combined = f"{question} {answer}"
            output_filename = os.path.join(output_folder, f"output_{idx}_{qa['id']}.txt")
            write_to_text_file(output_filename, context, qa_combined)
            print(f"Data written to {output_filename}")

if __name__ == "__main__":
    input_json_file = json_data  # Replace with your input JSON file path
    output_folder = txt_data  # Replace with the desired output folder path
    main(input_json_file, output_folder)
