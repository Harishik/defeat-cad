"""Tile still frames into a contact sheet:  python sheet.py out/stills/i_ out.png"""
import glob, sys
from PIL import Image
files = sorted(glob.glob(sys.argv[1] + "*.png"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
w, cols = 640, 3
ims = [Image.open(f).convert("RGB").resize((w, w * 9 // 16)) for f in files]
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * w * 9 // 16), "black")
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * w, (i // cols) * w * 9 // 16))
sheet.save(sys.argv[2])
print(len(ims), "frames")
