import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def convert_json_to_pdf(json_file, pdf_file):
    # Read JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a PDF document
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    elements = []

    # Define styles for Question and Text
    styles = getSampleStyleSheet()
    question_style = styles["Heading1"]
    text_style = styles["BodyText"]

    # Process each Q&A pair and add to the PDF
    for qa in data:
        question = qa["qas"][0].get("question", "")
        text = qa["qas"][0]["answers"][0].get("text", "")

        # Add Question to PDF
        elements.append(Paragraph("Question: {}".format(question), question_style))

        # Add Text to PDF
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Answer: {}".format(text), text_style))

        # Add a page break after each question and text
        elements.append(Spacer(1, 24))

    # Build the PDF document
    doc.build(elements)


if __name__ == "__main__":
    # Home-PC
    #path = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/Processing_Tools/Data_Set_Json_To_PDF_Converter/V2/"

    # Office
    path = r"D:/AI_CTS/Llama2/Processing_Tools/Data_Set_Json_To_PDF_Converter/V2/"


    input_json_file = path + "data_v2.json"  # Replace with the path to your input JSON file
    output_pdf_file = path + "Hotline_Wiki_v3.pdf"  # Replace with the desired output PDF file path
    convert_json_to_pdf(input_json_file, output_pdf_file)


