import PySimpleGUI as sg
import os
from PIL import Image
import io
from pathlib import Path
import cv2
import itertools
from collections import defaultdict
import numpy as np
'''
Hamming Distance - take two strings, compare differences. 
Equal image = 0 count (differences between scaled down images).
Usually around <5 indicates a duplicate/similar image but slightly modified. 


Hamming Distance no hash, I still do hash, sort of.
Convert image to grayscale, transforms down to 8x8, then compare.

aHash - average hash. 
Convert to grayscale, 8x8, average all pixel's colour,
use this average to convert image to 0 and 1's (below or above average). 
'''

#Take a list of images
#Maintain two lists of images
#1st list - Original images and a list of resized images used in hamming distance
#Resize the original images every time  and keep aspect ratio every time the user clicks on a row, to display it.
#Hamming distance algorithm:
#Convert to black and white
#Resize to 30x30
#Flatten the image. Optionally flatten it both row wise and column wise. 
#
#image = cv2.imread(args.file)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)  
#Compare two image strings. If they are idential hamming distance is 0. 
#If they differ by 1 value hamming distance is 1, etc... 

#Makes more sense to have an original table then a seperate table for each algorithm.

def define_window_layout():
    '''
    define_window_layout creates PySimpleGui window and layout.

    :return: A PySimpleGui.Window 
    '''
    def create_tree_layout(key_suffix):
        return [
            [sg.Tree(data=sg.TreeData(), headings=['Size', 'Dimensions', 'File Name'], 
                    auto_size_columns=False, num_rows=10, col0_width=100, key=f'-TREE{key_suffix}-',
                    enable_events=True)]
        ]

    # Create three separate tree layouts for each tab
    tree_layout1 = create_tree_layout('1')
    tree_layout2 = create_tree_layout('2')
    tree_layout3 = create_tree_layout('3')

    #Preview column layout (common for all tabs)
    preview_layout = [
        [sg.Text('Image Preview')],
        [sg.Image(key='-IMAGE-', size=(400, 300))]
    ]

    # Tab definitions
    tab1 = sg.Tab('Files', tree_layout1)
    tab2 = sg.Tab('Resized Images', tree_layout2)
    tab3 = sg.Tab('Hashed Image', tree_layout3)

    # Group tabs together
    tab_group = sg.TabGroup([[tab1, tab2, tab3]])

    # Combine the tab group and the preview into a single row
    combined_layout = [
        sg.Column([[tab_group]]),
        sg.VSeperator(),
        sg.Column(preview_layout, element_justification='center')
    ]

    #Update the main layout to include the combined layout
    layout = [
        [
            sg.Text('Included Directories'),
            sg.Button('Add', button_color=('white', 'green')),
            sg.Button('Remove', button_color=('white', 'red')),
            sg.Button('Manual Add')
        ],
        [
            sg.Text('Folders to Search'),
            sg.In(size=(25, 1), enable_events=True, key='-FOLDER-'),
            sg.FolderBrowse()
        ],
        [
            sg.HorizontalSeparator()
        ],
        [
            sg.Text('Resize algorithm'),
            sg.Combo(['Hamming Distance No Hash', 'Hamming Distance',], default_value='Hamming Distance No Hash', key='-RESIZE-'),
            sg.Text('Hash size'),
            sg.Spin([i for i in range(1, 33)], initial_value=8, key='-HASHSIZE-'),
            sg.Text('Hash type'),
            sg.Combo(['aHash', 'pHash', 'dHash'], default_value='Gradient', key='-HASHTYPE-'),
            sg.Checkbox('Search Directories Recursively', default=False, key='-RECURSIVE-'),
            sg.Button('Run Algorithm', expand_x=True),
        ],
        [
            sg.Text('Similarity'),
            sg.Slider(range=(0, 100), orientation='h', size=(20, 15), default_value=30, key='-SIMILARITY-')
        ],
        combined_layout,  #Insert the combined layout here
        [
            sg.Button('Search', expand_x=True), sg.Button('Select', expand_x=True),
            sg.Button('Sort', expand_x=True), sg.Button('Compare', expand_x=True),
            sg.Button('Delete', expand_x=True), sg.Button('Move', expand_x=True),
        ]
    ]

    #Create the Window with specified size
    return sg.Window('File Management System', layout, resizable=True)

def save_image(filename, directory_path, image):
    print(f"Saving image: {filename} to directory: {directory_path}")
    os.chdir(directory_path)
    resized = cv2.resize(image,(640, 640), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(filename, resized)

def save_dictionary_images(dictionary_images):
    for key, value in dictionary_images.items():
        save_image(value)

def rgb_to_grey(image):
    """
    rgb_to_grey Converts RGB numpy.ndarray to a grey numpy.ndarray.

    :param image: The numpy.ndarray image 
    :return: Grey image, numpy.ndarray
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_image_data(file_path, max_size=(400, 300)):
    '''
    get_image_data Opens an image from given a file_path,
    then resizes it, and returns the bytes.

    :param file_path: The path of the image to load. 
    :param max_size: The maximum size of an image to return.
    :return: The binary contents of the image, PNG file.
    '''
    try: 
        img = Image.open(file_path) #Opens file, doesn't load into memory, returns ~PIL.Image.Image
        img.thumbnail(max_size)
        bio = io.BytesIO() #In-memory binary stream. 
        img.save(bio, format="PNG") #Saves the image to the BytesIO object, bio now contains the binary data of the PNG image
        return bio.getvalue() #Returns the binary content of bio.
    except Exception as e:
        print(f'There was an error loading the image to display, error: {e}')
        return 'rut roh' #Return a blank error message later

def get_cv2_image(image_path):
    '''
    get_cv2_image Reads the image given a image_path.

    :param image_path: The path of the image to load
    :return: An image as a numpy.ndarray. 
    '''
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f'Error loading image: {e}\nFilepath: {image_path}')
        return 'rut roh' #Return a blank error message later

def resize(image, height=8, width=8):
    '''
    resize resizes an image given a height and width.

    :param image: An image in cv2 format
    :param height: Default 8, int determine's height of resized image 
    :param width: Default 8, int determine's width of resized image
    :return: Resized image as a numpy.ndarray
    '''
    resized = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA)
    return resized

def hamming_distance(image, image1):
    '''
    hamming_distance Calculates the hamming distance given two images (lists).

    :param image: The first image 
    :param image1: The second image
    :return: The hamming distance (a integer that represents how many differences were found in the list)
    '''
    print(f'Length: {len(image)}')
    print(f'Length1: {len(image1)}')
    count = sum(1 for a, b in zip(image, image1) if a != b)
    return count

#image_list -> Dict key=file_name_with_path value=flattened pixel values
def hamming_distance_naive(image_dictionary, threshold=10):
    '''
    hamming_distance_naive Calculates the hamming distance of every pair of images.

    :param image_list: A dictionary containing, key=file name with path, value=image, flattened numpy.ndarray.
    :param threshold: The hamming distance threshold value. < threshold means duplicate.
    :return: A dictionary containing the complete file paths of the duplicate images.  
    '''
    dict_duplicates = defaultdict(list)
    for k1, k2, in itertools.combinations(image_dictionary, 2):
        print(f'{k1}, {k2}, value: {hamming_distance(image_dictionary[k1], image_dictionary[k2])}')
        if hamming_distance(image_dictionary[k1], image_dictionary[k2]) < threshold:
            dict_duplicates[k1] = k2

    return dict_duplicates

def average_hash(image_dictionary):
    image_dictionary_averaged = {}
    for key, value in image_dictionary.items():
        #calculate average colour for image
        sum = 0
        image_length = value.size
        number_of_pixels = image_length
        for pixel in value: 
            sum += pixel
        average_colour = sum//number_of_pixels
        value = value > average_colour
        image_dictionary_averaged[key] = value
    return image_dictionary_averaged

def save_image_folder(image, filename, folder_name):
    cwd = os.getcwd()
    save_image(os.path.basename(filename), cwd+folder_name, image)
    os.chdir(cwd)

def black_white_resized_images(image_file_names):
    black_white_resized_images = {}
    for key, value in image_file_names.items():
        temp_image = (rgb_to_grey(value))
        temp = resize(temp_image)
        black_white_resized_images[key] = temp.flatten()
        save_image_folder(temp, key, '/resized-8x8')
    return black_white_resized_images

def main():
    hash_size = 8
    window = define_window_layout()
    image_file_names = {}

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        
        if event == '-TREE-':  # Event triggered when an item in the tree is clicked
            try:
                selected_file = window.Element('-TREE1-').SelectedRows[0]  # Get the index of the selected row
                print(f'4: {window.Element("-TREE1-").TreeData.tree_dict[selected_file].text}') 
                full_filename_path = window.Element("-TREE1-").TreeData.tree_dict[selected_file].text # text property of row index
                path = Path(full_filename_path)
                image_data = get_image_data(path)
                window['-IMAGE-'].update(image_data)
            except Exception as e:
                sg.popup_error('Error loading image:', e)

        elif event == 'Run Algorithm':
            algorithm = values['-RESIZE-']
            if image_file_names:
                if algorithm == 'Hamming Distance No Hash':
                    black_white_resized_images = black_white_resized_images(image_file_names)
                    dict_list = hamming_distance_naive(black_white_resized_images)
                    print(dict_list)

                elif algorithm == "Hamming Distance":
                    black_white_resized_images = black_white_resized_images(image_file_names)
                    images_averaged_hash_dict = average_hash(black_white_resized_images)
                    for key, value in images_averaged_hash_dict.items(): #Saving average_hash images. np.uint8 turns true/false array into 0,1. *255 from 0-1 to 0-255.
                        temp = np.uint8(value) * 255
                        save_image_folder(temp.reshape((hash_size,hash_size)) , key, '/average_hash-8x8')

                    dict_list = hamming_distance_naive(images_averaged_hash_dict)
                    print(dict_list)
                    
            else: 
                print('Error running algorithm, file list must not be empty!')

        elif event == 'Search':
            recursive = values['-RECURSIVE-']
            folder = values['-FOLDER-']

            file_names = search_directory(folder, recursive)
            for file_name in file_names: 
                image_file_names[file_name] = get_cv2_image(file_name)

            # Prepare data for the tree
            tree_data = sg.TreeData()
            for index, (key, value) in enumerate(image_file_names.items()):
                #print(index, key, value)
                tree_data.insert('', index, key, values=[f'{value.shape[:1]} x {value.shape[1:2]}', key])

            # Update the tree with the images
            window['-TREE1-'].update(values=tree_data)

    window.close()

'''
search_directory Gets the absolute path of all picture files given a directory, recursively or not. 

:param directory: (string) The directory from which to get the picture files from 
:param recursive: (boolean) True if recursively finding all filenames, false otherwise  
:return: A list with the absolute paths of all picture files from the given directory, recursively or not. 
'''
def search_directory(directory, recursive):
    file_names = []
    if recursive:
        try:
            for root, dirnames, files in os.walk(directory):
                for filename in files:
                    if filename.endswith(('.png', '.gif', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        file_names.append(os.path.join(root, filename))
        except Exception as e: 
            print(f"There was an error searching the directory, error: {e}")
    else:
        file_list = os.listdir(directory)
        try:
            # Get list of files and create dictionary 
            file_names = [
                os.path.join(directory, f) for f in file_list
                if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.tiff', '.bmp'))
            ]
        except Exception as e: 
            print(f"There was an error searching the directory, error: {e}")
    
    return file_names

if __name__ == '__main__':
    main()