# we only need to make system calls
import os

# all python files in this folder
os.system("autopep8 -r --in-place --aggressive ." )
os.system("docformatter --in-place -r .")
os.system("black -t py37 .")
os.system("flake8 .")