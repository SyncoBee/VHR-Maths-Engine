from PIL import Image, ImageDraw
import numpy as np

# Generate high-resolution 256x256 icon
size = 256
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Outer rounded plate
draw.rounded_rectangle([8, 8, 248, 248], radius=50, fill=(15, 23, 42, 255), outline=(56, 189, 248, 255), width=6)

# Generate mathematical spiral in the center
center_x, center_y = 128, 128
for t in np.linspace(0, 6 * np.pi, 600):
    r = 14 + 12.0 * t
    x = center_x + r * np.cos(t)
    y = center_y + r * np.sin(t)
    # Color gradient from cyan to electric purple to gold
    ratio = t / (6 * np.pi)
    red = int(56 * (1 - ratio) + 245 * ratio)
    green = int(189 * (1 - ratio) + 158 * ratio)
    blue = int(248 * (1 - ratio) + 11 * ratio)
    rad = 3.5 + 2.0 * ratio
    draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(red, green, blue, 240))

# Inner bright core
draw.ellipse([center_x - 12, center_y - 12, center_x + 12, center_y + 12], fill=(255, 255, 255, 255))

# Save multi-size .ico
import os
os.makedirs('assets', exist_ok=True)
img.save('assets/icon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save('assets/icon.png', format='PNG')
print('Generated assets/icon.ico and assets/icon.png successfully')
