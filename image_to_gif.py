import os
import imageio

# Directory containing the image files
image_dir = 'gif_frames'  # Replace with the path to your image directory

# List of image file extensions you want to include in the GIF
image_extensions = ['.jpg', '.png']  # Add more extensions as needed

# Initialize a list to store image file paths
image_files = []
image_arrays = []

# Iterate through files in the directory and collect image files
for filename in os.listdir(image_dir):
    if any(filename.endswith(ext) for ext in image_extensions):
        image_files.append(os.path.join(image_dir, filename))

# Sort the image files by name (you can customize the sorting logic if needed)
image_files.sort()
for image_file in image_files:
    image = imageio.imread(image_file)
    image_arrays.append(image)

# Specify the output GIF filename
output_filename = 'output.gif'

# Create the GIF using imageio
# with imageio.get_writer(output_filename, mode='I', duration=300.0) as writer:
#     for image_file in image_files:
#         image = imageio.imread(image_file)
#         writer.append_data(image)

imageio.mimsave(output_filename, image_arrays, format='GIF', duration=300)

print(f'GIF saved as {output_filename}')