import json

def json_to_markdown(json_data):
    markdown_text = ""
    for item in json_data:
        for qa in item.get("qas", []):
            question = qa.get("question", "")
            answers = [answer["text"] for answer in qa.get("answers", [])]
            markdown_text += f"## Question:  {question}\n\n"
            markdown_text += f"Answer: {answers[0]}\n\n"
    return markdown_text.strip()

def main():
    input_file = r"D:/AI_CTS/NLP/Processing_Tools/Data_Set_Json_To_MarkDown_Converter/train.json"  # Replace "input.json" with the path to your JSON file
    output_file = r"D:/AI_CTS/NLP/Processing_Tools/Data_Set_Json_To_MarkDown_Converter/output.md"  # Replace "output.md" with the desired output file name

    with open(input_file, "r") as file:
        json_data = json.load(file)

    markdown_text = json_to_markdown(json_data)

    with open(output_file, "w", encoding='utf-8') as file:  # Specify 'utf-8' encoding for writing
        file.write(markdown_text)

if __name__ == "__main__":
    main()