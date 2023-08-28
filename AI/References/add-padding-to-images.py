# Import modules
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add padding and center the image
def add_padding(image, background_color):
    # Get info
    width, height = image.size

    # Compare
    if width == height:
        return image
    elif width > height:
        image_padded = Image.new('RGB',
                                 (width, width),  # Width is longer -> width x width
                                 background_color)
        image_padded.paste(image, (0,                        # '0': x-coordinate of the top-left corner
                                  (width - height) // 2))   # 'width-height': difference between long and short edge
                                                            # '(width-height)//2': each for top and bottom
        return image_padded
    else:
        image_padded = Image.new('RGB',
                                 (height, height),
                                 background_color)
        image_padded.paste(image,
                           ((height - width) // 2,
                            0))
        return image_padded


def resize_with_padding(image, background_color):
    image = add_padding(image, background_color)
    width, height = image.size
    #print(width, height)
    image = image.resize((width, height), Image.ANTIALIAS)
    return image


image = Image.open('./data/car_license_plate3.PNG')
image2 = Image.open('./data/cat3.PNG')

image_new = resize_with_padding(image, (0, 0, 0))
image_new2 = resize_with_padding(image2, (0, 0, 0))

#plt.imshow(image)
#plt.show()

plt.imshow(image_new)
plt.show()

plt.imshow(image_new2)
plt.show()
