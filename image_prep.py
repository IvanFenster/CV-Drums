# pip install Pillow
from PIL import Image

image_file = "image/Background.png"
image = Image.open(image_file)
image = image.convert('RGBA')

w, h = image.size
pixdata = image.load()

pixel = []
n = 0
for x in range(w):
    for y in range(h):
        r, g, b, a = image.getpixel((x, y))
        print(r, g, b, a)
        if r < 40 and g < 40 and b < 40 and a > 100:
            pixdata[x, y] = (20, 20, 20, 255)
            n += 1

print(n)
image.save("output.png")