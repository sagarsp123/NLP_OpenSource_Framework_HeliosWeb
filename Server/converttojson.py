import csv
import json

def converttojson(csv_file_path):
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            json_data = {}

            for idx, row in enumerate(reader):
                sentence = row.get('Sentence', '')
                json_data[idx] = sentence

            return json_data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def write_json(json_data, txt_file_path):
    try:
        with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
            json.dump(json_data, txtfile, indent=4)
        print(f"JSON data successfully written to {txt_file_path}")
    except Exception as e:
        print(f"Error writing JSON to text file: {e}")

if __name__ == "__main__":
    csv_file_path = "test.csv"  # Replace with the path to your CSV file
    txt_file_path = "json.txt"  # Replace with the desired output text file path

    json_data = converttojson(csv_file_path)

    if json_data:
        write_json(json_data, txt_file_path)
