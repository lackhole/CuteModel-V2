import subprocess

try:
    # result = subprocess.run(['pwd'], stdout=subprocess.PIPE)
    # print(result.stdout.decode('utf-8'))
    
    # subprocess.run(['git', '--git-dir' 'tensorflow/.git', 'fetch origin'])
    subprocess.run(['git', 'submodule', 'update'])
    # subprocess.run(['git','submodule','update'])
    # subprocess.run(['cd', 'tensorflow'])
    
    print("Done")
    # print("Stdout: ", result.stdout.decode('utf-8'))
except Exception as E:
    print(E)
else:
    print("No Error")

# print("Stderr: ", result.stderr.decode('utf-8'))

