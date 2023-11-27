import PySimpleGUI as sg

# Define the layout for the window
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
    [
        sg.Tree(data=sg.TreeData(), headings=['Size', 'Dimensions', 'File Name'], 
                auto_size_columns=False, num_rows=10, col0_width=100, key='-TREE-')
    ],
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

    # Your event handling logic here

window.close()
