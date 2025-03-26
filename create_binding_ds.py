import csv

def extract_binding_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pdbid', '-logkd/ki'])  # Write header
        
        for line in infile:
            # Skip comment lines or empty lines
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdbid = parts[0]
                log_kd_ki = parts[3]
                writer.writerow([pdbid, log_kd_ki])

# Usage example
extract_binding_data('pdbbind_input.txt', 'binding_data.csv')
