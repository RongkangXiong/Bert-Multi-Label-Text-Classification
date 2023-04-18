# --coding:utf-8--
import sys

def filter_dependencies(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            if '@' in line:
                line = line.split('@')[0].strip() + '\n'
            f.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_dependencies.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    filter_dependencies(input_file, output_file)
