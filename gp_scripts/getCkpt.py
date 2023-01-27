import glob
import os
import pathlib
import sys

def get_Lastckpt():
    home_dir ='/users/PLS0150/gprasad' 
    list_of_files = glob.glob(home_dir + '/' +'scaleTrackML/logs/train/runs/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    logs = pathlib.Path(latest_file)
    list_ckpt=list(logs.rglob("epoch*.ckpt"))
    model_ckpt = str(list_ckpt[0])    
    return model_ckpt

if __name__ == "__main__":
    print(get_Lastckpt())
    sys.exit(0)