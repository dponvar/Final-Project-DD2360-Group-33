def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
    
    for i, (line1, line2) in enumerate(zip(f1_lines, f2_lines)):
        if line1 != line2:
            return f'Files differ at line {i+1}: {line1.strip()} != {line2.strip()}'
        else :
            return 'Files are identical'
# Example usage
file1 = 'newresult.txt'
file2 = 'oldresult.txt'
result = compare_files(file1, file2)
print(result)