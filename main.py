import PySimpleGUI as sg
import os
from PIL import Image
import io
from pathlib import Path
import cv2
import itertools
from collections import defaultdict
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

def hamming_distance(image1, image2):
    #Convert to strings

    #image_length

    #use itertools.combinations(dict, 2) to save computation. 
    #a, b, c, d, e   ab ac ad ae  avoid-> ba bc de instead-> bc bd be aka use itertools.combinations

    return

def rgb_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_image_data(file, max_size=(400, 300)):
    """
    Generate image data using PIL
    """
    img = Image.open(file)
    img.thumbnail(max_size)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def get_cv2_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f'Error loading image: {e}\nFilepath: {image_path}')
        return 'rut roh' #Return a blank error message later

def define_window_layout():
    # Define the layout for the window
    tree_layout = [
        [
            sg.Tree(data=sg.TreeData(), headings=['Size', 'Dimensions', 'File Name'], 
                    auto_size_columns=False, num_rows=10, col0_width=100, key='-TREE-',
                    enable_events=True)  # enable_events=True to trigger events when an item is clicked
        ]
    ]

    # Preview column layout
    preview_layout = [
        [sg.Text('Image Preview')],
        [sg.Image(key='-IMAGE-', size=(400, 300))]  # Placeholder for the image preview
    ]

    # Combine the tree and the preview into a single row
    combined_layout = [
        sg.Column(tree_layout),
        sg.VSeperator(),
        sg.Column(preview_layout, element_justification='center')
    ]

    # Update the main layout to include the combined layout
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
            sg.Combo(['Hamming Distance No Hash', 'Bilinear', 'Bicubic'], default_value='Hamming Distance No Hash', key='-RESIZE-'),
            sg.Text('Hash size'),
            sg.Spin([i for i in range(1, 33)], initial_value=16, key='-HASHSIZE-'),
            sg.Text('Hash type'),
            sg.Combo(['Gradient', 'Blockhash'], default_value='Gradient', key='-HASHTYPE-'),
            sg.Checkbox('Ignore same size', default=True, key='-IGNORESIZE-'),
            sg.Button('Run Algorithm', expand_x=True),
        ],
        [
            sg.Text('Similarity'),
            sg.Slider(range=(0, 100), orientation='h', size=(20, 15), default_value=30, key='-SIMILARITY-')
        ],
        combined_layout,  # Insert the combined layout here
        [
            sg.Button('Search', expand_x=True), sg.Button('Select', expand_x=True),
            sg.Button('Sort', expand_x=True), sg.Button('Compare', expand_x=True),
            sg.Button('Delete', expand_x=True), sg.Button('Move', expand_x=True),
        ]
    ]

    # Create the Window with specified size
    return sg.Window('File Management System', layout, resizable=True)

def resize(image, height=8, width=8):
    flattened = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    print(f'Flattened: {flattened.shape}')
    #flattened_column = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return flattened

def hamming_distance(image, image1):
    #count = 0
    #for index, pixel_value in enumerate(image): 
    #    if pixel_value != image1[index]:
    #        count += 1
    print(f'Length: {len(image)}')
    print(f'Length1: {len(image1)}')
    count = sum(1 for a, b in zip(image, image1) if a != b)
    return count
#Change image_list to a dictionary with the original filenames
#So when they are returned it will be easy to show what pictures are
def hamming_distance_naive(image_list, threshold=10):
    dict_duplicates = defaultdict(list)
    for k1, k2, in itertools.combinations(image_list, 2):
        print(f'{k1}, {k2}, value: {hamming_distance(image_list[k1], image_list[k2])}')
        if hamming_distance(image_list[k1], image_list[k2]) < threshold:
            dict_duplicates[k1] = k2

    return dict_duplicates

def main():
    window = define_window_layout()
    file_names = []
    image_file_names = {}

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        
        if event == '-TREE-':  # Event triggered when an item in the tree is clicked
            try:
                selected_file = window.Element('-TREE-').SelectedRows[0]  # Get the index of the selected row
                #print(f'1: {selected_file}')
                #print(f'2: {window.Element("-TREE-").TreeData}')
                #print(f'3: {window.Element("-TREE-").TreeData.tree_dict[selected_file].values[1]}')
                print(f'4: {window.Element("-TREE-").TreeData.tree_dict[selected_file].text}') 
                #print(window.Element('-TREE-').TreeData.tree_dict[selected_file])
                full_filename_path = window.Element("-TREE-").TreeData.tree_dict[selected_file].text # text property of row index
                path = Path(full_filename_path)
                image_data = get_image_data(path)
                window['-IMAGE-'].update(image_data)
            except Exception as e:
                sg.popup_error('Error loading image:', e)
        elif event == 'Run Algorithm':
            algorithm = values['-RESIZE-']
            if image_file_names:
                if algorithm == 'Hamming Distance No Hash':
                    black_white_resized_images = {}
                    for key, value in image_file_names.items():
                        temp_image = (rgb_to_grey(value))
                        black_white_resized_images[key] = (resize(temp_image))
                    dict_list = hamming_distance_naive(black_white_resized_images)

                    print(dict_list)

                    print('sneed')
            else: 
                print('Error running algorithm, file list must not be empty!')
        elif event == 'Search':
            folder = values['-FOLDER-']
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []
            # Get list of files and create dictionary 
            file_names = [
                os.path.join(folder, f) for f in file_list
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.tiff', '.bmp'))
            ]
            for file_name in file_names: 
                image_file_names[file_name] = get_cv2_image(file_name)

            # Prepare data for the tree
            tree_data = sg.TreeData()
            for index, (key, value) in enumerate(image_file_names.items()):
                #print(index, key, value)
                tree_data.insert('', index, key, values=[f'{value.shape[:1]} x {value.shape[1:2]}', key])

            # Update the tree with the images
            window['-TREE-'].update(values=tree_data)

    window.close()

if __name__ == '__main__':
    main()
