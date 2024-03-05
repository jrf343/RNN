import csv

def filter_philadelphia(input_file, output_file):
    with open(input_file, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        philadelphia_data = []
        for row in reader:
            if row['City'] == 'Philadelphia':
                philadelphia_data.append(row)

        with open(output_file, 'w', newline='') as csv_output:
            fieldnames = philadelphia_data[0].keys()
            writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(philadelphia_data)

input_file = 'GlobalLandTemperaturesByCity.csv'
output_file = 'PhiladelphiaLandTemperatures.csv'

filter_philadelphia(input_file, output_file)