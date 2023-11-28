import PySimpleGUI as sg
import os
from PIL import Image
import io
from pathlib import Path

def get_image_data(file, max_size=(400, 300)):
    """
    Generate image data using PIL
    """
    img = Image.open(file)
    img.thumbnail(max_size)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def main():
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
            sg.Combo(['Lanczos3', 'Bilinear', 'Bicubic'], default_value='Lanczos3', key='-RESIZE-'),
            sg.Text('Hash size'),
            sg.Spin([i for i in range(1, 33)], initial_value=16, key='-HASHSIZE-'),
            sg.Text('Hash type'),
            sg.Combo(['Gradient', 'Blockhash'], default_value='Gradient', key='-HASHTYPE-'),
            sg.Checkbox('Ignore same size', default=True, key='-IGNORESIZE-')
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
    window = sg.Window('File Management System', layout, resizable=True)

    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        
        if event == '-TREE-':  # Event triggered when an item in the tree is clicked
            try:
                # Assuming the tree data includes file paths, update the image preview
                selected_file = window.Element('-TREE-').SelectedRows[0]  # Get the file path of the selected item
                #print(f'1: {selected_file}')
                #print(f'2: {window.Element("-TREE-").TreeData}')
                #print(f'3: {window.Element("-TREE-").TreeData.tree_dict[selected_file].values[1]}')
                print(f'4: {window.Element("-TREE-").TreeData.tree_dict[selected_file].text}')
                #print(window.Element('-TREE-').TreeData.tree_dict[selected_file])
                full_filename_path = window.Element("-TREE-").TreeData.tree_dict[selected_file].text
                path = Path(full_filename_path)
                image_data = get_image_data(path)
                window['-IMAGE-'].update(image_data)
            except Exception as e:
                sg.popup_error('Error loading image:', e)

        elif event == 'Search':
            folder = values['-FOLDER-']
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            file_names = [
                os.path.join(folder, f) for f in file_list
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.tiff', '.bmp'))
            ]

                        # Prepare data for the tree
            tree_data = sg.TreeData()
            for index, filename in enumerate(file_names):
                #image_data = get_image_data(filename)
                tree_data.insert('', index, filename, values=['', filename])

            # Update the tree with the images
            window['-TREE-'].update(values=tree_data)

    window.close()

if __name__ == '__main__':
    main()
