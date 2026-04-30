import os
import zipfile
import fnmatch

EXCLUDES = [
    "data/*", "*/data/*", "proj5_code/*", "__pycache__/*", "*/__pycache__/*",
    "*.pyc", ".DS_Store", "*.zip", "*/.ipynb_checkpoints/*", "*.yml", 
    "*.tflite", "*.task", "assignment4_instructor/*"
]

def is_excluded(filepath):
    filepath = filepath.replace(os.sep, '/')
    for pattern in EXCLUDES:
        if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(os.path.basename(filepath), pattern):
            return True
        if pattern.endswith('/*') and filepath.startswith(pattern[:-2]):
            return True
    return False

def create_zip():
    out_zip = "assignment4_submission.zip"
    if os.path.exists(out_zip):
        os.remove(out_zip)
        
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk('.'):
            if root == '.':
                rel_root = ''
            else:
                rel_root = os.path.relpath(root, '.')
                
            for file in files:
                filepath = file if rel_root == '' else os.path.join(rel_root, file)
                if not is_excluded(filepath):
                    print(f"Adding: {filepath}")
                    zf.write(filepath)
                    
    print(f"\nSuccessfully created {out_zip}")

if __name__ == "__main__":
    create_zip()
