import os, sys, re

def main():
    src_dir = os.path.abspath('sim')
    warnings = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                # Look for any import of test modules or data loading from 'tests' folder
                if re.search(r'from\s+tests', content) or re.search(r'import\s+tests', content):
                    warnings.append(path)
    if warnings:
        print('Data leakage detected in the following files:')
        for w in warnings:
            print(' -', w)
        sys.exit(1)
    else:
        print('Data leakage check passed')
        sys.exit(0)

if __name__ == '__main__':
    main()
