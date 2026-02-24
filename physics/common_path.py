import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

# Folder with common data samples (<root>/sample_data)
COMMON_DATA_DIR = os.path.realpath(os.path.join(FILE_DIR, os.pardir, 'sample_data'))