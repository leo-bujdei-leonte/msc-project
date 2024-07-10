import os

def rename_increment(path: str, extension: str) -> str:
    if not os.path.isfile(path + "."  + extension):
        return path + "." + extension
    
    idx = 1
    while os.path.isfile(path + "_" + str(idx) + "." + extension):
        idx += 1
    
    return path + "_" + str(idx) + "." + extension