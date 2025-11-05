# prog7_compression.py
# pip install opencv-python pillow matplotlib
import cv2, heapq, matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

IMG = "images/image7.tif"
img = cv2.imread(IMG)

# ----- Huffman -----
class Node:
    def __init__(self,val,freq,left=None,right=None):
        self.val, self.freq, self.left, self.right = val,freq,left,right
    def __lt__(self, other): return self.freq < other.freq

def build_huffman(freq_map):
    pq = [Node(v,f) for v,f in freq_map.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        a, b = heapq.heappop(pq), heapq.heappop(pq)
        heapq.heappush(pq, Node(None, a.freq+b.freq, a, b))
    return pq[0]

def make_codes(node, code="", out=None):
    out = out or {}
    if node:
        if node.val is not None: out[node.val] = code
        make_codes(node.left,  code+"0", out)
        make_codes(node.right, code+"1", out)
    return out

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
freq = defaultdict(int)
for p in gray: freq[int(p)] += 1
root = build_huffman(freq)
codes = make_codes(root)
compressed_bits = "".join(codes[int(p)] for p in gray)
print("Huffman bit-length:", len(compressed_bits))

# ----- RLE -----
def rle_encode(arr):
    runs, prev, count = [], arr[0], 1
    for x in arr[1:]:
        if x==prev: count += 1
        else: runs.append((int(prev), count)); prev, count = x, 1
    runs.append((int(prev), count))
    return runs

rle = rle_encode(gray)
print("RLE tuples:", len(rle))

# ----- JPEG (lossy) -----
# (simplest: via PIL for quality control)
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pil.save("compressed.jpg", "JPEG", quality=80)
print("Saved JPEG: compressed.jpg (quality=80)")

# quick visual check
cmp = cv2.imread("compressed.jpg")
plt.figure(figsize=(9,4))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(cmp, cv2.COLOR_BGR2RGB)); plt.title("JPEG 80"); plt.axis('off')
plt.tight_layout(); plt.show()
