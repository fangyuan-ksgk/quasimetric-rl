import os
import subprocess

# Subprocess allows us to get the output from running a certain command
# -- this is actually one of the steps to tune LLM -- according to compilation errors
print('Directory Name checker: ')
cwd = os.path.dirname(__file__)
print(cwd)

# Output Checker
print('Subprocess Output Checker')
output = subprocess.check_output(
    r'git branch'.split(),
    cwd = os.path.dirname(__file__),
    encoding='utf-8'
).strip()
print(output)