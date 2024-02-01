from PIL import Image

# Open the PNG image
image = Image.open("/media/data/third_person_man/mask_test/Annotations/video1/00001.png")

# Convert to grayscale
gray_image = image.convert("L")

# Create a palette with a red color at index 1
palette = [0, 0, 0, 255,   # Index 0: Black
           255, 0, 0, 255]  # Index 1: Red

# Convert the image to use the palette
indexed_image = gray_image.convert("P", palette=Image.ADAPTIVE, colors=256)

# Save the result
indexed_image.save("/media/data/third_person_man/mask_test/Annotations/video1/00002.png")