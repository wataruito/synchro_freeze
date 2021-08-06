# Orignated from maximus009/VideoPlayer
# https://github.com/maximus009/VideoPlayer
"""
9/14/2020 wi Bug fix
                # freeze_end[epoch,i] = current_frame Identified bug 9/14/2020 wi
                freeze_end[epoch,i] = current_frame - 1

6/3/2021 wi add key
                   ord('a'): 'prev_frame', ord('d'): 'next_frame',
                   ord('A'): 'prev_frame_jump', ord('D'): 'next_frame_jump',
                   ord('~'): 'set_jump',
    command '~' will request new frame_jump nubmers
"""


import os
import sys
import time
import cv2
import numpy as np
import math
import pandas as pd

"""
Keyboard controls:
    <Video control>
    w: start palying
    s: stop playing
    a: step back a frame
    d: step forward a frame
    A: step back <frame_jump> frames
    D: step forward <frame_jump> frames
    ~: set <frame_jump> parameter
    q: play faster
    e: play slower

    <Tracking>
    0: drug mode
    1: sub1 click mode
    2: sub2 click mode

    <Freezing>
    !: target sub1
    @: target sub2
    j: start freezing
    k: end freezing
"""


def video_cursor(video, mag_factor):
    global x1, y1, x2, y2, drag, sub, click, mode, pixel_limit

    ###################################
    # Initialize video windows

    cv2.namedWindow('image')
    # cv2.moveWindow('image',250,150)
    # Set mouse callback
    cv2.setMouseCallback('image', dragging)
    # Open video file
    cap = cv2.VideoCapture(video)
    # Get the total number of frame
    tots = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Add two trackbars
    cv2.createTrackbar('S', 'image', 0, int(tots)-1, flick)
    cv2.setTrackbarPos('S', 'image', 0)

    cv2.createTrackbar('F', 'image', 1, 100, flick)
    frame_rate = 30
    cv2.setTrackbarPos('F', 'image', frame_rate)
    # cv2.setTrackbarPos('F','image',0)
    ##################################
    # Initialize freeze indicator window for each subject

    sub_freeze = ['sub1_freeze', 'sub2_freeze']

    cv2.namedWindow(sub_freeze[0])
    cv2.moveWindow(sub_freeze[0], 250, 50)

    cv2.namedWindow(sub_freeze[1])
    cv2.moveWindow(sub_freeze[1], 600, 50)

    # Create new blank image
    # freeze
    width, height = 200, 50
    red = (255, 0, 0)
    freeze_sign = create_blank(width, height, rgb_color=red)
    cv2.putText(freeze_sign, "Freeze", (40, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)
    # no_freeze
    green = (0, 255, 0)
    no_freeze_sign = create_blank(width, height, rgb_color=green)
    cv2.putText(no_freeze_sign, "No_freeze", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)

    ###################################

    status_list = {ord('s'): 'stop',
                   ord('w'): 'play',
                   ord('a'): 'prev_frame', ord('d'): 'next_frame',
                   ord('A'): 'prev_frame_jump', ord('D'): 'next_frame_jump',
                   ord('~'): 'set_jump',
                   ord('q'): 'slow', ord('e'): 'fast',
                   ord(' '): 'snap',
                   ord('0'): 'drag_mode', ord('1'): 'click_mode_sub1', ord('2'): 'click_mode_sub2',
                   ord('!'): 'target_sub1', ord('@'): 'target_sub2',
                   ord('j'): 'start_freezing', ord('k'): 'end_freezing',
                   -1: 'no_key_press',
                   27: 'exit'}
    current_frame = 0
    frame_jump = 10
    status = 'stop'
    start_time = time.time()
    realFrameRate = frame_rate

    # Adjust video width 750 pixel
    ret, im = cap.read()
    video_format = im.shape
    print("Video resolution: {}".format(video_format))
    x_pixcels = im.shape[1]
    y_pixcels = im.shape[0]
    # r = 3
    dim = (x_pixcels*mag_factor, y_pixcels*mag_factor)

    print("total frame number: {}".format(tots))

    length = 5  # cross cursor length

    # Mouse events handler (global)
    x1, y1, x2, y2 = 100, 100, 120, 100
    drag = False
    click = False
    sub = ''
    pixel_limit = 10.0
    mode = 'drag_mode'

    # prepare to store freezing
    target_freeze = -1
    freeze_state = False
    freeze_modify = False
    freeze_modify_on_frame = -1
    freeze_modify_off_frame = -1

    # prepare to store trajectory and freezing
    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'

    if os.path.exists(os.path.join(path, filename)):
        # xy1, xy2, freeze = read_trajectory(video)
        width, halfDep, L1, L2, L4, xy1, xy2, freeze = read_traj(video)
    else:
        width = 295.0
        halfDep = 86.5
        L1 = [-5, 667]
        L2 = [42, 486]
        L4 = [914, 670]
        xy1 = np.array([[-1 for x in range(2)] for y in range(tots)])
        xy2 = np.array([[-1 for x in range(2)] for y in range(tots)])
        freeze = np.array([[False for x in range(2)] for y in range(tots)])

    ######################################################################
    # Main loop
    while True:
        try:
            # If reach to the end, play from the begining
            # if current_frame==tots-1:
            # if current_frame == tots:
            if current_frame >= tots:
                current_frame = 0

            # read a frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, im = cap.read()

            # resize video
            im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

            # put current state and real frame rate in the image
            im_text1 = "video_status: " + status + ", frame_rate: " + \
                str(realFrameRate) + " fps"
            im_text2 = "nmode: " + mode + \
                ", target_freeze: " + str(target_freeze+1) + \
                ", freeze_modify: " + str(freeze_modify)

            add_text(im, im_text1, dim[1]-40, 0.5)
            add_text(im, im_text2, dim[1]-20, 0.5)

            ###################################
            # display cursors
            if (xy1[current_frame, 0] == -1 or drag or click) and sub == 'sub1':
                xy1[current_frame, :] = [x1, y1]
            [x1, y1] = xy1[current_frame, :]
            # [_x1,_y1] = xy1[current_frame,:]
            cv2.line(im, (x1+length, y1+length),
                     (x1-length, y1-length), (0, 255, 0), 2)
            cv2.line(im, (x1+length, y1-length),
                     (x1-length, y1+length), (0, 255, 0), 2)

            if (xy2[current_frame, 0] == -1 or drag or click) and sub == 'sub2':
                xy2[current_frame, :] = [x2, y2]
            [x2, y2] = xy2[current_frame, :]
            # [_x2,_y2] = xy2[current_frame,:]
            cv2.line(im, (x2+length, y2+length),
                     (x2-length, y2-length), (0, 0, 255), 2)
            cv2.line(im, (x2+length, y2-length),
                     (x2-length, y2+length), (0, 0, 255), 2)

            ###################################
            # display freezing state

            if freeze_modify_on_frame == current_frame:
                if target_freeze == -1:
                    freeze_modify_on_frame = -1
                else:
                    freeze_modify = True

            if freeze_modify_off_frame == current_frame:
                if freeze_modify == True:
                    freeze_modify = False
                    for i in range(freeze_modify_on_frame, freeze_modify_off_frame + 1):
                        freeze[i, target_freeze] = True

                    freeze_modify_on_frame = -1
                    freeze_modify_off_frame = -1

            for i in range(2):
                if freeze_modify and target_freeze == i:
                    # print(i)
                    cv2.imshow(sub_freeze[i], freeze_sign)
                elif freeze[current_frame, i] == True:
                    cv2.imshow(sub_freeze[i], freeze_sign)
                else:
                    cv2.imshow(sub_freeze[i], no_freeze_sign)

            ###################################
            # show video frame
            cv2.imshow('image', im)

            ###################################
            # Read key input
            status_new = status_list[cv2.waitKey(1)]

            if status_new != 'no_key_press':
                status_pre = status
                status = status_new

            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F', 'image')
                if frame_rate == 0.0:
                    continue
                if (time.time() - start_time) > 1.0/frame_rate:
                    realFrameRate = round(1.0/(time.time() - start_time), 2)
                    current_frame += 1
                    cv2.setTrackbarPos('S', 'image', current_frame)
                    start_time = time.time()
                    continue
            elif status == 'stop':
                current_frame = cv2.getTrackbarPos('S', 'image')
            elif status == 'prev_frame_jump':
                # current_frame-=1
                current_frame = current_frame - frame_jump
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'next_frame_jump':
                # current_frame+=1
                current_frame = current_frame + frame_jump
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'set_jump':
                frame_jump = int(input())
                print('frame_jump is set as: ', frame_jump, ' frames')
                status = status_pre
            elif status == 'prev_frame':
                current_frame -= 1
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'next_frame':
                current_frame += 1
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'slow':
                frame_rate = max(frame_rate - 1, 0)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status = status_pre
            elif status == 'fast':
                frame_rate = min(100, frame_rate+1)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status = status_pre
            elif status == 'drag_mode':
                mode = 'drag_mode'
                status = status_pre
            elif status == 'click_mode_sub1':
                mode = 'click_mode_sub1'
                status = status_pre
            elif status == 'click_mode_sub2':
                mode = 'click_mode_sub2'
                status = status_pre
            elif status == 'target_sub1':
                target_freeze = 0
                status = status_pre
            elif status == 'target_sub2':
                target_freeze = 1
                status = status_pre
            elif status == 'start_freezing':
                # freeze_state = True
                freeze_modify_on_frame = current_frame
                status = status_pre
            elif status == 'end_freezing':
                # freeze_state = False
                freeze_modify_off_frame = current_frame
                status = status_pre
            elif status == 'snap':
                cv2.imwrite("./"+"Snap_"+str(i)+".jpg", im)
                print("Snap of Frame", current_frame, "Taken!")
                status = 'stop'
            elif status == 'exit':
                break

        except KeyError:
            print("Invalid Key was pressed")
    # Clean up windows
    cap.release()
    cv2.destroyAllWindows()

    # write file for trajectory and freezing
    # write_trajectory(tots,xy1,xy2,freeze,video)
    write_traj(width, halfDep, L1, L2, L4, tots, xy1, xy2, freeze, video)

    # write file for freeze start, end duration
    write_freeze(tots, freeze, video)

    return

##################################################################################################


def write_freeze(tots, freeze, video):

    epoch_max = 100

    freeze_start = np.array([[-1 for x in range(2)]
                            for y in range(epoch_max)], dtype=int)
    freeze_end = np.array([[-1 for x in range(2)]
                          for y in range(epoch_max)], dtype=int)
    freeze_dur = np.array([[-1.0 for x in range(2)]
                          for y in range(epoch_max)], dtype=float)
    epoch_n = [0, 0]

    for i in range(2):
        freeze_on = False
        epoch = -1
        for current_frame in range(tots):
            if freeze[current_frame, i] == True and freeze_on == False:    # freeze epoch starts
                epoch += 1
                freeze_start[epoch, i] = current_frame
                freeze_on = True
            elif freeze[current_frame, i] == False and freeze_on == True:  # freeze ephoc ends
                # freeze_end[epoch,i] = current_frame Identified bug 9/14/2020 wi
                freeze_end[epoch, i] = current_frame - 1
                freeze_dur[epoch, i] = (
                    freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
                freeze_on = False
        epoch_n[i] = epoch
        if freeze_on:
            freeze_end[epoch, i] = current_frame
            freeze_dur[epoch, i] = (
                freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
            freeze_on = False

    columnName = ['start', 'end', 'duration', 'start', 'end', 'duration']
    columnType = ['int',   'int', 'float',    'int',   'int', 'float']

    df = pd.DataFrame(
        data=np.concatenate((freeze_start[:, 0][:, np.newaxis], freeze_end[:, 0][:, np.newaxis], freeze_dur[:, 0][:, np.newaxis],
                             freeze_start[:, 1][:, np.newaxis], freeze_end[:, 1][:, np.newaxis], freeze_dur[:, 1][:, np.newaxis]), axis=1),
        columns=columnName)

    a = dict(zip(columnName, columnType))
    df = df.astype(dtype=dict(zip(columnName, columnType)))

    # Output to summary.csv
    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_freeze.csv'

    print("\tWriting {}".format(filename))
    write_pd2csv(path, filename, df, columnName, columnType, 1000)

    return


##################################################################################################
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

##################################################################################################


def read_trajectory(video):
    import os

    columnName = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
                  'sub2_y', 'sub1_freeze', 'sub2_freeze']
    columnType = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']

    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'

    print("\tReading {}".format(filename))

    df = read_csv2pd(path, filename, columnName, columnType)
    # xy1 = [df['sub1_x'], df['sub1_y']]
    xy1 = df[['sub1_x', 'sub1_y']].to_numpy()
    xy2 = df[['sub2_x', 'sub2_y']].to_numpy()
    freeze = df[['sub1_freeze', 'sub2_freeze']].to_numpy()

    return(xy1, xy2, freeze)


def read_csv2pd(path, filename, columnName, columnType):
    import os
    import numpy as np
    import pandas as pd

    inputFilename = os.path.join(path, filename)

    df = pd.read_csv(inputFilename, index_col=False)
    # Need to convert to object to set numpy array in a cell
    df = df.astype(object)

    # Post process from str to array
    for i in range(0, len(df)):
        for j in range(0, len(df.columns)):
            if columnType[j] == 'int_array':
                df.iloc[i, j] = np.fromstring(
                    df.iloc[i, j][1:-1], dtype=int, sep=' ')

    return(df)

##################################################################################################


def write_trajectory(tots, xy1, xy2, freeze, video):
    # Initialize Pandas DataFrame
    columnName = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
                  'sub2_y', 'sub1_freeze', 'sub2_freeze']
    columnType = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']
    frame_num = np.array([y for y in range(tots)])[
        np.newaxis]  # Need 2D matrix to tranpose
    frame_num = np.transpose(frame_num)
    df = pd.DataFrame(data=np.concatenate(
        (frame_num, xy1, xy2, freeze), axis=1), columns=columnName)
    a = dict(zip(columnName, columnType))
    df = df.astype(dtype=dict(zip(columnName, columnType)))

    # Output to summary.csv
    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'

    print("\tWriting {}".format(filename))
    write_pd2csv(path, filename, df, columnName, columnType, 1000)

    return


def write_pd2csv(path, filename, df, columnName, columnType, mlw=1000):
    import os
    import numpy as np
    import pandas as pd

    outputFilename = os.path.join(path, filename)
    output = open(outputFilename, "w")
    # mlw = 1000 # max_line_width in np.array2string

    output.write(','.join(columnName)+'\n')

    for i in range(0, len(df)):
        output_str = ''
        for j in range(0, len(columnName)):
            # print(df.shape,j)
            output_str = preprocess_output_str(
                output_str, df.iloc[i, j], columnType[j], 1000)
        output.write(output_str[0:-1] + '\n')
    output.close()
    return


def preprocess_output_str(output_str, data, columnType, mlw=1000):
    import numpy as np

    if columnType == 'int_array':
        output_str = output_str + \
            np.array2string(data, max_line_width=mlw) + ','
    # elif columnType == 'float':
    elif columnType == 'float' or columnType == 'bool':
        # print(data)
        output_str = output_str + str(data) + ','
    elif columnType == 'str':
        output_str = output_str + data + ','
    elif columnType == 'int':
        output_str = output_str + str(data) + ','
    return(output_str)

###################################
# Mouse events handler


def dragging(event, x, y, flags, param):
    global x1, y1, x2, y2, drag, sub, click, mode, pixel_limit
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'drag_mode':
            if abs(math.sqrt(x1**2+y1**2) - math.sqrt(x**2+y**2)) < pixel_limit:
                drag = True
                sub = 'sub1'
                x1, y1 = x, y
            if abs(math.sqrt(x2**2+y2**2) - math.sqrt(x**2+y**2)) < pixel_limit:
                drag = True
                sub = 'sub2'
                x2, y2 = x, y
        elif mode[0:-1] == 'click_mode_sub':
            click = True
            if mode[-1] == '1':
                x1, y1 = x, y
            if mode[-1] == '2':
                x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False
        sub = ''
        click = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag == True:
            if sub == 'sub1':
                x1, y1 = x, y
            elif sub == 'sub2':
                x2, y2 = x, y

###################################


def flick(x):
    pass


def process(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def add_text(img, text, text_top, image_scale):
    """
    Args:
        img (numpy array of shape (width, height, 3): input image
        text (str): text to add to image
        text_top (int): position of top text to add
        image_scale (float): image resize scale

    Summary:
        Add display text to a frame.

    Returns:
        Next available position of top text (allows for chaining this function)
    """
    cv2.putText(
        img=img,
        text=text,
        org=(0, text_top),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=image_scale,
        color=(0, 255, 255),
        thickness=2)
    return text_top + int(5 * image_scale)
##################################################################################################


def read_traj(video):
    # read *_trac_freeze.csv file and extract
    #    landmark coordinates for L1, L2, and L4
    #    trajectory coordinates for sub1 and sub2
    #    freezing state (bool) for sub1 and sub2
    #
    # <file format>
    # measurement:
    # L1-L4(width), 295.0
    # L1-L2(halfDep), 86.5
    #
    # landmark:
    # name,x,y
    # L1, ,
    # L2, ,
    # L4, ,
    #
    # coordinate:
    # frame,sub1_x,sub1_y,sub2_x,sub2_y,sub1_freeze,sub2_freeze
    #
    # Old format, which starts with frame,sub1_x ... can be read.
    #

    import csv
    import os
    import numpy as np
    import pandas as pd

    columnName = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
                  'sub2_y', 'sub1_freeze', 'sub2_freeze']
    columnType = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']

    # defalt values
    width = 295.0
    halfDep = 86.5
    L1, L2, L4 = [0, 0], [0, 0], [0, 0]

    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'
    inputFilename = os.path.join(path, filename)

    print("\tReading {}".format(filename))

    # reading csv file
    with open(inputFilename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            if row[0] == 'L1-L4(width)':
                width = float(row[1])
            elif row[0] == 'L1-L2(halfDep)':
                halfDep = float(row[1])
            elif row[0] == 'L1':
                L1 = [int(row[1]), int(row[2])]
            elif row[0] == 'L2':
                L2 = [int(row[1]), int(row[2])]
            elif row[0] == 'L4':
                L4 = [int(row[1]), int(row[2])]
            elif row[0] == 'coordinate:':
                break
            elif row[0] == 'frame':
                csvfile.seek(csvreader.line_num - 1)  # back one line
                break
        # after break, use dataframe.read_csv
        df = pd.read_csv(csvfile, index_col=False)
        # Need to convert to object to set numpy array in a cell
        df = df.astype(object)

    # Post process from str to array
    for i in range(0, len(df)):
        for j in range(0, len(df.columns)):
            if columnType[j] == 'int_array':
                df.iloc[i, j] = np.fromstring(
                    df.iloc[i, j][1:-1], dtype=int, sep=' ')

    xy1 = df[['sub1_x', 'sub1_y']].to_numpy()
    xy2 = df[['sub2_x', 'sub2_y']].to_numpy()
    freeze = df[['sub1_freeze', 'sub2_freeze']].to_numpy()

    return(width, halfDep, L1, L2, L4, xy1, xy2, freeze)

##################################################################################################


def write_traj(width, halfDep, L1, L2, L4, tots, xy1, xy2, freeze, video):
    # write *_trac_freeze.csv file
    #
    # <file format>
    # measurement:
    # L1-L4(width), 295.0
    # L1-L2(halfDep), 86.5
    #
    # landmark:
    # name,x,y
    # L1, ,
    # L2, ,
    # L4, ,
    #
    # coordinate:
    # frame,sub1_x,sub1_y,sub2_x,sub2_y,sub1_freeze,sub2_freeze
    #

    import os
    import csv
    import numpy as np
    import pandas as pd

    # Initialize Pandas DataFrame
    columnName = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
                  'sub2_y', 'sub1_freeze', 'sub2_freeze']
    columnType = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']
    frame_num = np.array([y for y in range(tots)])[
        np.newaxis]  # Need 2D matrix to tranpose
    frame_num = np.transpose(frame_num)
    df = pd.DataFrame(data=np.concatenate(
        (frame_num, xy1, xy2, freeze), axis=1), columns=columnName)
    df = df.astype(dtype=dict(zip(columnName, columnType)))

    # Output to summary.csv
    path, filename = os.path.split(video)
    base, ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'
    outputFilename = os.path.join(path, filename)

    print("\tWriting {}".format(filename))

    with open(outputFilename, 'w', newline='') as csvfile:  # newline='' is for windows
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['measurement:'])
        spamwriter.writerow(['L1-L4(width)', width])
        spamwriter.writerow(['L1-L2(halfDep)', halfDep])
        spamwriter.writerow([''])
        spamwriter.writerow(['landmark:'])
        spamwriter.writerow(['name', 'x', 'y'])
        spamwriter.writerow(['L1', L1[0], L1[1]])
        spamwriter.writerow(['L2', L2[0], L2[1]])
        spamwriter.writerow(['L4', L4[0], L4[1]])
        spamwriter.writerow([''])
        spamwriter.writerow(['coordinate:'])

        df.to_csv(csvfile, index=False)

    return
##################################################################################################
