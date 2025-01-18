import ROOT
from pathlib import Path

CPP_DIR = Path(__file__).parent / 'Lorenzetti'
CPP_FILE_EXTENSIONS = [
    '*.h'
]
for file_ext in CPP_FILE_EXTENSIONS:
    for filename in CPP_DIR.glob(file_ext):
        print(f'Loading file {filename}')
        text = filename.read_text()
        ROOT.gInterpreter.Declare(text)

del CPP_DIR
del CPP_FILE_EXTENSIONS
