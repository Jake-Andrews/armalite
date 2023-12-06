import PySimpleGUI as sg
import os
from PIL import Image
import io
from pathlib import Path, PurePath
import cv2
import itertools
from collections import defaultdict
import numpy as np
from sys import getsizeof
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
            [sg.Tree(data=sg.TreeData(), headings=['Size (Bytes)', 'Dimensions', 'File Name'], 
                    auto_size_columns=False, num_rows=10, col0_width=0, col_widths=[25,25,80], key=f'-TREE{key_suffix}-',
                    enable_events=True)]
        ]

    # Create three separate tree layouts for each tab
    tree_layout1 = create_tree_layout('1')
    tree_layout2 = create_tree_layout('2')
    tree_layout3 = create_tree_layout('3')
    tree_layout4 = create_tree_layout('4')

    #Preview column layout (common for all tabs)
    preview_layout = [
        [sg.Text('Image Preview')],
        [sg.Image(key='-IMAGE-', size=(400, 300))]
    ]

    # Tab definitions
    tab1 = sg.Tab('Images', tree_layout1)
    tab2 = sg.Tab('Resized Images', tree_layout2)
    tab3 = sg.Tab('Hashed Images', tree_layout3)
    tab4 = sg.Tab('Duplicate Images', tree_layout4)

    # Group tabs together
    tab_group = sg.TabGroup([[tab1, tab2, tab3, tab4]])

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
            sg.Button('Clear Intermediate Tabs', ),
            sg.Button('Clear All Tabs')
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

    :param image_dictionary: A dictionary containing, key=filename with path, value=image, non-flattened numpy.ndarray.
    :param threshold: The hamming distance threshold value. < threshold means duplicate.
    :return: A dictionary containing the complete file paths of the duplicate images.  
    '''
    dict_duplicates = defaultdict(list)
    for k1, k2, in itertools.combinations(image_dictionary, 2):
        print(f'{k1}, {k2}, value: {hamming_distance(image_dictionary[k1].flatten(), image_dictionary[k2].flatten())}')
        if hamming_distance(image_dictionary[k1].flatten(), image_dictionary[k2].flatten()) < threshold:
            dict_duplicates[k1].append(k2)

    return dict_duplicates

def average_hash(image_dictionary):
    image_dictionary_averaged = {}
    for key, value in image_dictionary.items():
        #Calculate average colour for image
        sum = 0
        value = value.flatten()
        image_length = value.size
        number_of_pixels = image_length
        for pixel in value: 
            sum += pixel
        #Apply average colour mask (true if > avg colour, false otherwise)
        average_colour = sum//number_of_pixels
        value = value > average_colour
        image_dictionary_averaged[key] = value
    return image_dictionary_averaged

def save_image_folder(black_white_resized_images_dict, folder_name):
    cwd = os.getcwd()
    for filepath, image in black_white_resized_images_dict.items():
        save_image(os.path.basename(filepath), cwd+folder_name, image)
    os.chdir(cwd)

def black_white_resize_images(image_file_names):
    black_white_resized_images = {}
    for key, value in image_file_names.items():
        temp_image = (rgb_to_grey(value))
        black_white_resized_images[key] = resize(temp_image)
    return black_white_resized_images

def main():
    hash_size = 8
    window = define_window_layout()
    image_file_names = {}

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        def load_image(tree_number):
            try:
                selected_file = window.Element(f'-TREE{tree_number}-').SelectedRows[0] #Get the index of the selected row
                print(f'4: {window.Element(f"-TREE{tree_number}-").TreeData.tree_dict[selected_file].values[2]}') 
                full_filename_path = window.Element(f"-TREE{tree_number}-").TreeData.tree_dict[selected_file].values[2] #Text property of row index
                path = Path(full_filename_path)
                image_data = get_image_data(path)
                window['-IMAGE-'].update(image_data)
            except Exception as e:
                sg.popup_error('Error loading image:', e)

        if event == '-TREE1-': #Event triggered when an item in the tree is clicked
            load_image(1)
        elif event == '-TREE2-': 
            load_image(2)
        elif event == '-TREE3-':
            load_image(3)
        elif event == '-TREE4-':
            load_image(4)

        elif event == 'Clear Intermediate Tabs':
            for i in range(2,5):
                window[f'-TREE{i}-'].update(values=sg.TreeData())
        elif event == 'Clear All Tabs':
            for i in range(1,5):
                window[f'-TREE{i}-'].update(values=sg.TreeData())
            image_file_names = {}

        elif event == 'Run Algorithm':
            algorithm = values['-RESIZE-']
            if image_file_names:
                if algorithm == 'Hamming Distance No Hash':
                    black_white_resized_images = black_white_resize_images(image_file_names)
                    
                    #Save the resized black and white images, then display them in a tab
                    save_image_folder(black_white_resized_images, '/resized-8x8')
                    tree_data = prepare_image_dict_for_tree(black_white_resized_images, 'resized-8x8')
                    window['-TREE2-'].update(values=tree_data)
                    
                    dict_list = hamming_distance_naive(black_white_resized_images)
                    print(dict_list)
                    tree_data_2 = prepare_image_dict_for_tree(dict_list, '', spacing=True)
                    window['-TREE4-'].update(values=tree_data_2)

                elif algorithm == "Hamming Distance":
                    black_white_resized_images = black_white_resize_images(image_file_names)
                    
                    #Save the resized black and white images, then display them in a tab
                    save_image_folder(black_white_resized_images, '/resized-8x8')
                    tree_data = prepare_image_dict_for_tree(black_white_resized_images, 'resized-8x8')
                    window['-TREE2-'].update(values=tree_data)
                    
                    images_averaged_hash_dict = average_hash(black_white_resized_images)

                    new_temp_rounded_dict = {k:(np.uint8(v) * 255).reshape((hash_size,hash_size)) for k, v in images_averaged_hash_dict.items()}
                    #Save the averaged hash images, then display them in a tab
                    save_image_folder(new_temp_rounded_dict, '/average_hash-8x8')
                    tree_data_1 = prepare_image_dict_for_tree(new_temp_rounded_dict, 'average_hash-8x8')
                    window['-TREE3-'].update(values=tree_data_1)

                    dict_list = hamming_distance_naive(images_averaged_hash_dict)
                    print(dict_list)
                    tree_data_2 = prepare_image_dict_for_tree(dict_list, '', spacing=True)
                    print(f"TreeLdata: {tree_data_2}")
                    window['-TREE4-'].update(values=tree_data_2)
                    
            else: 
                print('Error running algorithm, file list must not be empty!')

        elif event == 'Search':
            recursive = values['-RECURSIVE-']
            folder = values['-FOLDER-']
            #Grab images from folders, then load the images into dict
            file_names = search_directory(folder, recursive)
            if file_names:
                for file_name in file_names: 
                    image_file_names[file_name] = get_cv2_image(file_name)
                #Prepare data for the tree
                tree_data = prepare_image_dict_for_tree(image_file_names)
                window['-TREE1-'].update(values=tree_data)
            else: print("Please enter a valid directory!")

    window.close()

def prepare_image_dict_for_tree(image_dictionary, insert_directory='', spacing=False):
    #Prepare data for the tree
    print(insert_directory)
    tree_data = sg.TreeData()
    #...:( 
    if insert_directory:
        for index, (key, value) in enumerate(image_dictionary.items()):
            print(key)
            temp = PurePath(key)
            temp_parts = list(temp.parts)
            print(temp_parts)

            temp_parts.insert(-1, insert_directory)
            new_path = PurePath('').joinpath(*temp_parts)
            print(new_path)
            tree_data.insert('', index, '', values=[getsizeof(value), f'{value.shape[:1]} x {value.shape[1:2]}', new_path])        
    else:
        index = 0
        for key, value in image_dictionary.items():
            print(f"key: {key}, value: {value}")
            #in this case image_dictionary is defaultdict. key=filename, value=filename list. 
            if type(value) == list and spacing==True:
                #First row is the key (image used to compare against others)
                image = get_cv2_image(key) 
                tree_data.insert('', index, '', values=[getsizeof(image), f'{image.shape[:1]} x {image.shape[1:2]}', key])
                index += 1
                #Next rows are the values (images detected to be duplicates of the above)
                for filename in value:
                    image = get_cv2_image(filename) 
                    tree_data.insert('', index, '', values=[getsizeof(image), f'{image.shape[:1]} x {image.shape[1:2]}', filename])
                    index += 1
                #create blank row to seperate duplicate images visually
                tree_data.insert('', index, '', values=['', '', ''])
                index += 1

            #image_dictionary is normal dict. key=filename, value=image (numpy array)
            else: 
                tree_data.insert('', index, '', values=[getsizeof(value), f'{value.shape[:1]} x {value.shape[1:2]}', key])
                index += 1
    return tree_data

def search_directory(directory, recursive):
    '''
    search_directory Gets the absolute path of all picture files given a directory, recursively or not. 

    :param directory: (string) The directory from which to get the picture files from 
    :param recursive: (boolean) True if recursively finding all filenames, false otherwise  
    :return: A list with the absolute paths of all picture files from the given directory, recursively or not. 
    '''
    if not directory:
        return ''
    
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
        try:
            file_list = os.listdir(directory)
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