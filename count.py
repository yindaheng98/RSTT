import glob
import re
for path in glob.glob('experiments/RSTT-S_archived_220705-162106/models/*.log'):
    with open(path, "r") as f:
        print(path)
        s = f.read()
        m = re.search(r"Total Average PSNR: ([0-9.]+?) dB", s)
        print(m.group(1))