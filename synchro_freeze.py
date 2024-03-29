###############################################################################
# pipeline to process epochs from two subjects
# The original script started computer freezing epoch, but extended to generic epochs.
#
'''
File format for the input csv file:
Currently start and end are in frame number and duration is in sec. Need to unify them.

dir:                20210711
old_dir:            na
exp_id:             m274_1
sex:                male
age:                96
infusion_hpc:       muscimol
infusion_pfc:
familiarity:        familiar
lighting:           visible
partition:          FALSE
stress:             no_stress
video:              HomeCage
video_total_frames: 17998
comment:            dorsal HPC

data:   start   end     duration    start   end     duration
        303     308     1.5         435     440     1.5
        347     352     1.5         444     451     2
        354     359     1.5         627     631     1.25
        371     375     1.25        682     686     1.25
        405     409     1.25        700     704     1.25
        413     419     1.75
        434     445     3
        447     454     2
        480     501     5.5
        633     645     3.25
comment:	total (s)	71.5						79
			%			59.58333333					65.83333333
end:
'''
###############################################################################
import synchro_freeze as sf
import random
import sys
import traceback
from sre_constants import IN_IGNORE
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def process_freeze(path, DEBUG):

    # import matplotlib.pyplot as plt
    # import os
    # import numpy as np
    # import pandas as pd

    # DEBUG = 0

    #############################################
    # Step 1. Create master summary.csv file
    #   Read the input CSV file containing epoch onset and offset from two subjects
    #   Create Pandas DataFrame
    #   Output summary.csv
    # The format of the pandas dataframe and output csv is specified by colunName and columnType
    #############################################
    print("Step1. Reading CSV files from subfolders.")
    columnName = ['folder_videoname',   'single_animal',    'video_system',     'video_total_frames',
                  'fz_start_sub1',      'fz_end_sub1',      'fz_start_sub2',    'fz_end_sub2']
    columnType = ['str',                'bool',             'str',              'int',
                  'int_array',          'int_array',        'int_array',        'int_array']
    create_master_table(path, columnName, columnType)

    #############################################
    # Step 2. Create trajectory summary
    #   Read the input '_track_freeze.csv file containing trajectories from two subjects
    #   Create Pandas DataFrame
    #   Output summary_traj.csv
    # Since no further process so far, the foramt specification is inside of the function.
    #############################################
    print("Step2. Reading trajectory CSV files from subfolders.")
    create_trajectory_table(path)

    #############################################
    # Step 3. Compute % epoch time and add the summary
    #   Read summary.csv
    #   Compute % freezing and store in DF
    #   Output to summary1.csv
    # Add the three columns, as indicated with columnName and columnType
    #############################################
    print("Step3. Computing % epoch time.")
    columnName = np.append(columnName, ['fz_sub1',  'fz_sub2',  'fz_overlap'])
    columnType = np.append(columnType, ['float',    'float',    'float'])
    compute_epoch_percent(path, columnName, columnType)

    #############################################
    # Step 4. Compute Cohen_D
    #   Read summary1.csv
    #   Compute permutation/Cohen_D and store in DF
    #   Output to summary2.csv
    #############################################
    print("Step4. Computing permutation/Cohen_D and store in DF.")
    columnName = np.append(columnName, ['cohen_d'])
    columnType = np.append(columnType, ['float'])
    compute_cohen_d(path, columnName, columnType)

    #############################################
    # Step 5. Markov analysis
    #   Read summary3.csv
    #   Count behavioral state & transitions for Markov chain analysis
    #   Output to summary3.csv
    #############################################
    print("Step5. Counting behavioral state & stansitions for Markov chain analysis")
    columnName = np.append(columnName, ['s_count_0', 's_count_1', 's_count_2', 's_count_3',
                                        'st_count_00', 'st_count_01', 'st_count_02', 'st_count_03',
                                        'st_count_10', 'st_count_11', 'st_count_12', 'st_count_13',
                                        'st_count_20', 'st_count_21', 'st_count_22', 'st_count_23',
                                        'st_count_30', 'st_count_31', 'st_count_32', 'st_count_33'])
    columnType = np.append(columnType, ['int', 'int', 'int', 'int',
                                        'int', 'int', 'int', 'int',
                                        'int', 'int', 'int', 'int',
                                        'int', 'int', 'int', 'int',
                                        'int', 'int', 'int', 'int'])
    compute_markov_chain(path, columnName, columnType)

    # 2021/10/4 wi: The follwing parts still generate error. Need fix.
    #############################################
    # Read summary2.csv
    # Compute lag times
    # Output to summary3.csv
    #############################################
    # compute_lagtime()

    #############################################
    # Read summary4.csv
    # Compute averaged distance during CS and store in DF
    # Output to summary5.csv
    #############################################
    # compute_distance()

    #############################################
    # Debugging commands
    if DEBUG:
        print(df.dtypes)
        print(df)
        for i in range(0, len(df)):
            for j in range(0, len(df.columns)):
                print(type(df.iloc[i, j]))

    if not DEBUG:
        filename = os.path.join(path, 'summary.csv')
        os.remove(filename)
        filename = os.path.join(path, 'summary1.csv')
        os.remove(filename)
        filename = os.path.join(path, 'summary2.csv')
        os.remove(filename)
        filename = os.path.join(path, 'summary3.csv')
        os.remove(filename)
        filename = os.path.join(path, 'summary4.csv')
        os.remove(filename)
    return(df, df_traj)


###############################################################################
# create_master_table()
#       _read_csv()
#       write_pd2csv()
#           preprocess_output_str()
def create_master_table(path, columnName, columnType, output_file='summary1.csv'):

    # Initialize Pandas DataFrame for master table
    df = pd.DataFrame()
    # columnName = ['folder_videoname', 'single_animal', 'video_system', 'video_total_frames',
    #               'fz_start_sub1', 'fz_end_sub1', 'fz_start_sub2', 'fz_end_sub2']
    # columnType = ['str', 'bool', 'str', 'int',
    #               'int_array', 'int_array', 'int_array', 'int_array']
    for i in range(len(columnName)):
        df[columnName[i]] = []

    # Search subfolders, read input csv files, and append DF
    # print("Step1. Reading CSV files from subfolders.")
    print("\tProcessing directory: ", end=" ")
    for dir_name in os.listdir(path):
        if dir_name[0:1] != "_":
            path1 = os.path.join(path, dir_name)
            if os.path.isdir(path1):
                print("{}, ".format(dir_name), end=" ")
                for file in os.listdir(path1):
                    base = os.path.splitext(file)[0]
                    extn = os.path.splitext(file)[1]

                    # if it is master csv file
                    if extn == '.csv' and base[0] != '_':
                        base1 = ' '*13 + base

                        if base1[-13:] != '_track_freeze':
                            filename = os.path.join(path1, file)
                            # Read CSV file
                            # print(filename)
                            single_animal, video_system, video_total_frames, sub1Start, sub1End, sub2Start, sub2End = \
                                _read_csv(filename)

                            # Store in PD dataframe
                            # 20220331 wi replaced pd.append to pd.concat
                            # Make sure use [] blacket to put each array into a single cell.
                            dir_name_base = dir_name + '_' + base
                            _df = {
                                columnName[0]: dir_name_base,
                                columnName[1]: single_animal,
                                columnName[2]: video_system,
                                columnName[3]: video_total_frames,
                                columnName[4]: [sub1Start],
                                columnName[5]: [sub1End],
                                columnName[6]: [sub2Start],
                                columnName[7]: [sub2End]
                            }
                            _df = pd.DataFrame.from_dict(_df)
                            df = pd.concat([df, _df], ignore_index=True)

    print("completed.")

    # Output to summary.csv
    print("\tWriting " + output_file + ".")
    write_pd2csv(path, output_file, df, columnName, columnType, 1000)


def _read_csv(filename):
    # Read original csv file of start and stop of each freezing epoch

    # importing csv module
    import csv

    # csv file name
    # filename = "/home/wito/Dropbox/Jupyter/20190207_old_females_sync_freezing - female pair1.csv"
    # filename = r"C:\Users\User\Desktop\project\20200419-134553\20190207_old_females_sync_freezing - female pair1.csv"

    # initializing the titles and rows list
    fields1 = []
    fields2 = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    _sub1Start = []
    _sub1End = []
    _sub2Start = []
    _sub2End = []

    # Detect single or pair freezing
    single_animal = 'FALSE'
    for i in range(0, len(rows)):
        if rows[i][0] == 'single_animal:':
            single_animal = rows[i][1]

    # Detect video system, either FreezeFrame or PCBox
    video_system = ''
    for i in range(0, len(rows)):
        if rows[i][0] == 'video:':
            video_system = rows[i][1]

    # Detect video_total_frames.
    video_total_frames = 0
    for i in range(0, len(rows)):
        if rows[i][0] == 'video_total_frames:':
            video_total_frames = rows[i][1]

    # Read start and end for subject1
    # for i in range(2,len(rows)):
    _data_start = False
    for i in range(0, len(rows)):
        if rows[i][0] == 'data:':
            _data_start = True
        elif _data_start:
            if rows[i][1] == '':
                break
            # print(i)
            _sub1Start.append(int(rows[i][1]))
            _sub1End.append(int(rows[i][2]))

    # Read start and end for subject2
    _data_start = False
    for i in range(0, len(rows)):
        if rows[i][0] == 'data:':
            _data_start = True
        elif _data_start:
            if rows[i][4] == '':
                break
            _sub2Start.append(int(rows[i][4]))
            _sub2End.append(int(rows[i][5]))

    # convert to numpy array
    import numpy as np
    sub1Start = np.array(_sub1Start)
    sub1End = np.array(_sub1End)
    sub2Start = np.array(_sub2Start)
    sub2End = np.array(_sub2End)

    return (single_animal, video_system, video_total_frames, sub1Start, sub1End, sub2Start, sub2End)


def write_pd2csv(path, filename, df, columnName, columnType, mlw=1000):
    import os
    import numpy as np
    import pandas as pd

    outputFilename = os.path.join(path, filename)
    output = open(outputFilename, "w")
    # mlw = 1000 # max_line_width in np.array2string

    # output column labels
    output.write(','.join(columnName)+'\n')

    # output each row
    for i in range(0, len(df)):
        # print(i)
        output_str = ''
        # Assemble a single row
        for j in range(0, len(columnName)):
            # print(df.shape,j)
            # output_str = preprocess_output_str(
            #     output_str, df.iloc[i, j], columnType[j], 1000)
            # output_str = preprocess_output_str(
            #     output_str, df[columnName[j]][i], columnType[j], mlw)
            output_str = output_str + \
                preprocess_output_str1(
                    df[columnName[j]][i], columnType[j], mlw) + ','
        output.write(output_str[0:-1] + '\n')
    output.close()
    return


def preprocess_output_str1(data, columnType, mlw=1000):
    import numpy as np

    if columnType == 'int_array':
        output_str = np.array2string(data, max_line_width=mlw)
    elif columnType == 'float' or columnType == 'bool' or columnType == 'int':
        output_str = str(data)
    elif columnType == 'str':
        output_str = data

    return(output_str)


# def preprocess_output_str(output_str, data, columnType, mlw=1000):
#     import numpy as np

#     if columnType == 'int_array':
#         output_str = output_str + \
#             np.array2string(data, max_line_width=mlw) + ','
#     elif columnType == 'float' or columnType == 'bool' or columnType == 'int':
#         output_str = output_str + str(data) + ','
#     elif columnType == 'str':
#         output_str = output_str + data + ','

#     return(output_str)


###############################################################################
# create_trajectory_table()
#       read_traj()
#       toRealCordinate()
#       write_pd2csv()
def create_trajectory_table(path):

    # Initialize Pandas DataFrame for trajectory table
    # create dataframe for real-world coordinate
    columnName_traj_sub = list(map(str, list(range(0, 722))))
    columnName_traj = ['id', 'sub_id'] + columnName_traj_sub
    columnType_traj = ['str', 'str'] + ['float']*722
    df_traj = pd.DataFrame(columns=columnName_traj)

    # Search subfolders, read trajectory csv files, and append DF
    # print("Step2. Reading trajectory CSV files from subfolders.")
    print("\tProcessing directory: ", end=" ")
    for dir_name in os.listdir(path):
        if dir_name[0:1] != "_":
            path1 = os.path.join(path, dir_name)
            if os.path.isdir(path1):
                print("{}, ".format(dir_name), end=" ")
                for file in os.listdir(path1):
                    base = os.path.splitext(file)[0]
                    extn = os.path.splitext(file)[1]

                    # if it is master csv file
                    if extn == '.csv' and base[0] != '_':
                        base1 = ' '*13 + base

                        # if it is trajectory csv file
                        if base1[-13:] == '_track_freeze':
                            filename = os.path.join(path1, file)
                            # Read trajectory file (*_track_freeze.csv)
                            width, halfDep, L1, L2, L4, xy1, xy2, freeze = read_traj(
                                filename)

                            # convert pixel coordinate to real-world coordinate
                            sub = np.array([[0 for y in range(len(xy1))]
                                           for x in range(4)], dtype=float)
                            for i in range(len(xy1)):
                                sub[0, i], sub[1, i] = toRealCordinate(
                                    xy1[i, :], L1, L2, L4, width, halfDep)
                                sub[2, i], sub[3, i] = toRealCordinate(
                                    xy2[i, :], L1, L2, L4, width, halfDep)

                            # append computed data to df
                            dir_name_base = dir_name + '_' + \
                                base.replace('_track_freeze', '')
                            # pair_id = 'test1-1' # supposed to be set as 'dir_name_base'
                            sub_ids = ['sub1_x', 'sub1_y', 'sub2_x', 'sub2_y']
                            for i in range(4):
                                df_1 = pd.DataFrame(
                                    [[dir_name_base, sub_ids[i]]], columns=['id', 'sub_id'])
                                # df_2 = pd.DataFrame([sub[i,:]], columns=columnName_traj_sub)
                                df_2 = pd.DataFrame([sub[i, :]], columns=list(
                                    map(str, list(range(0, len(xy1))))))
                                df_3 = pd.concat([df_1, df_2], axis=1)
                                df_traj = pd.concat(
                                    [df_traj, df_3], ignore_index=True)

    print("completed.")

    # Output to summary.csv
    print("\tWriting summary_traj.csv.")
    write_pd2csv(path, 'summary_traj.csv', df_traj,
                 columnName_traj, columnType_traj, 1000)


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
    # base,ext = os.path.splitext(filename)
    # filename = '_' + base + '_track_freeze.csv'
    # inputFilename = os.path.join(path,filename)
    inputFilename = video

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


def toRealCordinate(sub, p1, p2, p4, width, halfDep):
    # convert pixel coordinate to real-world coordinate

    sub = sub.copy()
    p1 = p1.copy()
    p2 = p2.copy()
    p4 = p4.copy()

    sub[1] = 1000 - sub[1]
    p1[1] = 1000 - p1[1]
    p2[1] = 1000 - p2[1]
    p4[1] = 1000 - p4[1]

    # x_length
    y_diff = p4[1]-p1[1]
    x_diff = p4[0]-p1[0]
    cos_sin = abs(x_diff)*abs(y_diff)/(x_diff**2 + y_diff**2)
    inter1_x = sub[0]
    inter1_y = p1[1] + y_diff*(sub[0]-p1[0])/x_diff

    if p4[1] > p1[1]:
        inter2_x = inter1_x + (sub[1] - inter1_y) * cos_sin
    else:
        inter2_x = inter1_x - (sub[1] - inter1_y) * cos_sin

    x_length = width * (inter2_x - p1[0]) / x_diff

    # y_length
    y12_diff = p2[1]-p1[1]
    x12_diff = p2[0]-p1[0]
    y14_diff = p4[1]-p1[1]
    x14_diff = p4[0]-p1[0]

    import math

    sin_a = abs(y14_diff) / math.sqrt(x14_diff**2 + y14_diff**2)
    cos_a = abs(x14_diff) / math.sqrt(x14_diff**2 + y14_diff**2)
    sin_b = abs(y12_diff) / math.sqrt(x12_diff**2 + y12_diff**2)
    cos_b = abs(x12_diff) / math.sqrt(x12_diff**2 + y12_diff**2)

    inter1_y = sub[1]
    inter1_x = p1[0] + x12_diff * (sub[1] - p1[1]) / y12_diff

    a = sub[0] - inter1_x
    b = a * sin_a

    len_1_2 = math.sqrt(x12_diff**2 + y12_diff**2)
    len_1_inter1 = math.sqrt((inter1_x - p1[0])**2 + (inter1_y - p1[1])**2)

    # if p1[0] < p2[0] and p1[1] < p4[1]:-
    # if p1[0] < p2[0] and p1[1] > p4[1]:+
    # if p1[0] > p2[0] and p1[1] < p4[1]:-
    # if p1[0] > p2[0] and p1[1] > p4[1]:+

    if p1[1] > p4[1]:
        sin_g = sin_b*cos_a + cos_b*sin_a
        c = b / sin_g
        len_1_inter2 = len_1_inter1 + c
    else:
        sin_g = sin_b*cos_a - cos_b*sin_a
        c = b / sin_g
        len_1_inter2 = len_1_inter1 - c

    y_length = halfDep / len_1_2 * len_1_inter2

    return x_length, y_length


###############################################################################
# compute_epoch_percent()
#       read_csv2pd()
#       overlap_freezing()
#       write_pd2csv()
def compute_epoch_percent(path, columnName, columnType, input_file='summary.csv', output_file='summary1.csv'):

    # Read CSV into pandas DF
    df = read_csv2pd(path, input_file, columnName, columnType)

    # Compute % freezing and store in DF
    # print("\nStep3. Computing %_epoch_time.")

    # columnName = np.append(columnName, ['fz_sub1', 'fz_sub2', 'fz_overlap'])
    # columnType = np.append(columnType, ['float', 'float', 'float'])

    sub1Freeze = np.zeros(len(df))
    sub2Freeze = np.zeros(len(df))
    overlapFreeze = np.zeros(len(df))

    print("\tProcessing:  ", end=" ")

    for i in range(0, len(df)):
        if i % 1000 == 0:
            print(i, '/', len(df), end=', ')

        subfolder = os.path.join(path, df.iloc[i, 0])
        (sub1Freeze[i], sub2Freeze[i], overlapFreeze[i], overlap) = overlap_freezing(
            df.iloc[i, :], subfolder, False)
        if df['single_animal'][i] == 'TRUE':
            sub2Freeze[i] = "nan"
            overlapFreeze[i] = "nan"

    # Add columns & data
    _df = pd.DataFrame()
    # i = 8
    _df[columnName[-3]] = sub1Freeze
    _df[columnName[-2]] = sub2Freeze
    _df[columnName[-1]] = overlapFreeze
    df = df.join(_df)

    # Output to summary1.csv
    print("\tWriting " + output_file + ".")
    write_pd2csv(path, output_file, df, columnName, columnType, 1000)


def read_csv2pd(path, filename, columnName, columnType):
    import os
    import numpy as np
    import pandas as pd

    inputFilename = os.path.join(path, filename)

    df = pd.read_csv(inputFilename, index_col=False)
    # Need to convert object to set numpy array in a cell
    # df = df.astype(object)

    # Post process to convert from str to int_array
    for i in range(0, len(df)):
        for j in range(len(columnName)):
            if columnType[j] == 'int_array':
                df.iat[i, df.columns.get_loc(columnName[j])] = np.fromstring(
                    df.loc[i, columnName[j]][1:-1], dtype=int, sep=' ')

    # Adjust dtypes according columnType
    type_dict = {columnName[i]: columnType[i] for i in range(len(columnName))}
    # Select only columns that exist in the df
    type_dict = {key: type_dict[key] for key in type_dict.keys() & df.columns}
    df = df.astype(
        {name: np.int64 for name, age in type_dict.items() if age == 'int'})
    df = df.astype(
        {name: np.float64 for name, age in type_dict.items() if age == 'float'})

    return(df)


def read_csv2pd_org(path, filename, columnName, columnType):
    import os
    import numpy as np
    import pandas as pd

    inputFilename = os.path.join(path, filename)

    df = pd.read_csv(inputFilename, index_col=False)
    # Need to convert object to set numpy array in a cell
    df = df.astype(object)

    # Post process from str to array
    for i in range(0, len(df)):
        for j in range(0, len(df.columns)):
            if columnType[j] == 'int_array':
                # Fix bug for error 2021/07/13 wi
                # "ValueError: Must have equal len keys and value when setting with an iterable"
                # df.iloc[i, j] = np.fromstring(
                #    df.iloc[i, j][1:-1], dtype=int, sep=' ')
                df.iat[i, j] = np.fromstring(
                    df.iloc[i, j][1:-1], dtype=int, sep=' ')

    return(df)


###############################################################################
# compute_epoch_percent2()
#       read_csv2pd()
#       overlap_freezing3()
#           frames_range()
#       write_pd2csv()
def compute_epoch_percent3(path, columnName, columnType, input_file='summary2.csv', output_file='summary3.csv'):
    import warnings
    warr_lines = []

    # Read CSV into pandas DF
    df = read_csv2pd(path, input_file, columnName, columnType)

    # Compute % freezing and store in DF
    # print("\nStep3. Computing %_epoch_time.")

    # columnName = np.append(columnName, ['fz_sub1', 'fz_sub2', 'fz_overlap'])
    # columnType = np.append(columnType, ['float', 'float', 'float'])

    sub1Freeze = np.zeros(len(df))
    sub2Freeze = np.zeros(len(df))
    overlapFreeze = np.zeros(len(df))

    fz_sub1_bout_dur_ave = np.zeros(len(df))
    fz_sub1_bout_n = np.zeros(len(df))
    fz_sub2_bout_dur_ave = np.zeros(len(df))
    fz_sub2_bout_n = np.zeros(len(df))

    print("\tProcessing:  ", end=" ")

    for i in range(0, len(df)):
        if i % 10 == 0:
            print(i, '/', len(df), end=', ')

        subfolder = os.path.join(path, df.iloc[i, 0])

        # (sub1Freeze[i], sub2Freeze[i], overlapFreeze[i], overlap,
        #     fz_sub1_bout_dur_ave[i], fz_sub1_bout_n[i], fz_sub2_bout_dur_ave[i], fz_sub2_bout_n[i]) \
        #     = overlap_freezing3(df.iloc[i, :], subfolder, False)

        # If freezing is zero, the old overlap_freezing2 shows warning message
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                (sub1Freeze[i], sub2Freeze[i], overlapFreeze[i], overlap,
                 fz_sub1_bout_dur_ave[i], fz_sub1_bout_n[i], fz_sub2_bout_dur_ave[i], fz_sub2_bout_n[i]) \
                    = overlap_freezing3(df.iloc[i, :], subfolder, False)
            except Warning:
                print('Raised! ', '##', i,
                      df['folder_videoname'][i], '## ', end='')
                warr_lines = np.append(warr_lines, df['folder_videoname'][i])

        if df['single_animal'][i] == True:
            # sub2Freeze[i] = 0
            # overlapFreeze[i] = 0
            # fz_sub2_bout_dur_ave[i] = 0
            # fz_sub2_bout_n[i] = 0
            sub2Freeze[i] = np.nan
            overlapFreeze[i] = np.nan
            fz_sub2_bout_dur_ave[i] = np.nan
            # We need to set value 0 at integer because we cannot set nan as integer
            fz_sub2_bout_n[i] = 0

    # Add columns & data
    _df = pd.DataFrame()
    # i = 8
    _df[columnName[-7]] = sub1Freeze
    _df[columnName[-6]] = sub2Freeze
    _df[columnName[-5]] = overlapFreeze
    _df[columnName[-4]] = fz_sub1_bout_dur_ave
    _df[columnName[-3]] = fz_sub1_bout_n
    _df[columnName[-2]] = fz_sub2_bout_dur_ave
    _df[columnName[-1]] = fz_sub2_bout_n

    df = df.join(_df)

    # Output to summary1.csv
    print("\tWriting " + output_file + ".")
    write_pd2csv(path, output_file, df, columnName, columnType, 1000)

    return warr_lines


def frames_range(video_system, video_total_frames):
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    # 1) FreezeFrame generates video 0-721 frames (total 722 frames)
    # Only care 1-720 frames (total 720 frames), ignoring frame# 0 (black frame) and 721.
    #     1 min for acclimation (frame 1-240)
    #     2 min for CS (frame 241-720))
    #
    #     Create np.array of 721 rows x 2 columes,
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     row 0 will be not used.

    # PCbox generates video 0-719 frames (total 720 frames)
    """
    Extended to generic epoch from home cage videos

    Parenthesis indicates the video frames to be compute for % epoch time
    FreezeFrame 0,  (pre: 1 - 240), (CS: 241 - 720), 721,   Total 722 frames
    PCBox           (pre: 0 - 239), (CS: 240 - 719),        Total 720 frames
    HomeCage - analyze the entire frames
                    (0 to video_total_frames - 1),          Total video_total_frames frames
    """

    if video_system == 'FreezeFrame':
        # FreezeFrame
        csStartFrame, csEndFrame = 241, 720
        startFrame, endFrame = 0, 721
    if video_system == 'FreezeFrame_1st_half':
        # FreezeFrame
        csStartFrame, csEndFrame = 241, 480
        startFrame, endFrame = 0, 721
    if video_system == 'FreezeFrame_2nd_half':
        # FreezeFrame
        csStartFrame, csEndFrame = 481, 720
        startFrame, endFrame = 0, 721
    if video_system == 'FreezeFrame_contextual':
        # FreezeFrame
        csStartFrame, csEndFrame = 1, 720
        startFrame, endFrame = 0, 721
    if video_system == 'PCBox':
        # PCBox
        csStartFrame, csEndFrame = 240, 719
        startFrame, endFrame = 0, 719
    if video_system == 'HomeCage':
        # Generic, presumably home cage annotation
        csStartFrame, csEndFrame = 0, video_total_frames - 1
        startFrame, endFrame = 0, video_total_frames - 1

    return csStartFrame, csEndFrame, startFrame, endFrame


def overlap_freezing3(df, path, output):
    # Compute overlap of freezing
    #

    import matplotlib.pyplot as plt
    import os
    import numpy as np

    filename = os.path.join(path, 'overlap_fig.eps')

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    csStartFrame, csEndFrame, startFrame, endFrame = frames_range(
        video_system, video_total_frames)  # Get the frames range

    # Generate bit map for freezing in the range from csStartFrame to csEndFrame
    # Initialize the bit map
    overlap = np.array([[0 for x in range(3)] for y in range(endFrame + 1)])

    # For animal#1
    # initialize duration of each freezing bout
    fz_sub1_bout_dur = np.zeros(len(sub1Start))

    for i in range(0, int(len(sub1Start))):
        for j in range(sub1Start[i], sub1End[i] + 1):
            if j >= csStartFrame and j <= csEndFrame:
                overlap[j][0] = 1
                fz_sub1_bout_dur[i] += 1
        # If fz_sub1_bout_dur == 0, the bout is outside of the range
        if fz_sub1_bout_dur[i] == 0:
            fz_sub1_bout_dur[i] = np.nan
    # Remove bouts outside of the range
    fz_sub1_bout_dur = fz_sub1_bout_dur[~np.isnan(fz_sub1_bout_dur)]

    # For animal#2
    # initialize duration of each freezing bout
    fz_sub2_bout_dur = np.zeros(len(sub2Start))

    for i in range(0, int(len(sub2Start))):
        for j in range(sub2Start[i], sub2End[i] + 1):
            if j >= csStartFrame and j <= csEndFrame:
                overlap[j][1] = 1
                fz_sub2_bout_dur[i] += 1
        if fz_sub2_bout_dur[i] == 0:
            fz_sub2_bout_dur[i] = np.nan
    fz_sub2_bout_dur = fz_sub2_bout_dur[~np.isnan(fz_sub2_bout_dur)]

    # Count each numbers
    #   Freezing in animal#1: counter[0]
    #   Freezing in animal#2: counter[1]
    #   Overlapped freezing:  counter[2]
    counter = np.zeros((3), dtype=int)

    for i in range(csStartFrame, csEndFrame + 1):
        if overlap[i, 0] == 1:
            counter[0] = counter[0] + 1
        if overlap[i, 1] == 1:
            counter[1] = counter[1] + 1
        if overlap[i, 0] == 1 and overlap[i, 1] == 1:
            overlap[i, 2] = 1
            counter[2] = counter[2] + 1

    # For animal#1
    data_range = float(csEndFrame - csStartFrame + 1)
    sub1Freeze = counter[0]/data_range*100.0
    fz_sub1_bout_dur_ave = np.average(fz_sub1_bout_dur/4.0)
    fz_sub1_bout_n = len(fz_sub1_bout_dur)

    # For animal#2
    # only process animal#2 when 'single_animal' is False
    if df['single_animal'] != True:
        sub2Freeze = counter[1]/data_range*100.0
        fz_sub2_bout_dur_ave = np.average(fz_sub2_bout_dur/4.0)
        fz_sub2_bout_n = len(fz_sub2_bout_dur)

        overlapFreeze = counter[2]/data_range*100.0
    else:
        sub2Freeze = "nan"
        fz_sub2_bout_dur_ave = "nan"
        fz_sub2_bout_n = "nan"

        overlapFreeze = "nan"

    # output
    if output:
        print("Folder name: " + df[0])
        print("Animal1 freeze : %f" % (sub1Freeze))
        print("Animal2 freeze : %f" % (sub2Freeze))
        print("Overlap freeze : %f" % (overlapFreeze))

        # Plotting the freezing dynamics
        fig = plt.figure(num=None, figsize=(15, 5), dpi=80,
                         facecolor='w', edgecolor='k')
        fig.subplots_adjust(top=0.8)

        ax1 = fig.add_subplot(211)
        x = overlap[:, 0] + 1.75
        y = overlap[:, 1] + 1.25
        z = overlap[:, 2]

        ax1.plot(x)
        ax1.plot(y)
        ax1.plot(z)

        ax1.set_xlabel('x - axis')
        ax1.set_ylabel('y - axis')
        ax1.set_title('From top: Animal1, Animal2 and overlap!')
        plt.savefig(filename, format='eps', dpi=1000)

    return(sub1Freeze, sub2Freeze, overlapFreeze, overlap, fz_sub1_bout_dur_ave, fz_sub1_bout_n, fz_sub2_bout_dur_ave, fz_sub2_bout_n)


###############################################################################
# (p_)compute_cohen_d()        p_ for multiprocessing
#       read_csv2pd()
#       (p_)permutation()
#           overlap_freezing()
#       write_pd2csv()
def compute_cohen_d(path, columnName, columnType, input_file='summary2.csv', output_file='summary3.csv'):
    # Read CSV into pandas DF
    df = read_csv2pd(path, input_file, columnName, columnType)

    # Compute permutation/Cohen_D and store in DF
    print("\tProcessing column: ", end=" ")
    Cohen_D = np.zeros(len(df))

    for i in range(0, len(df)):
        if i % 10 == 0:
            print("{}/{}, ".format(i, len(df)), end=" ")

        if df['single_animal'][i] == True or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            # print("hit!")
            Cohen_D[i] = "nan"
        else:
            subfolder = os.path.join(path, df.iloc[i, 0])
            Cohen_D[i] = permutation(df.iloc[i, :], subfolder, False)
    print("completed.")

    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[-1]] = Cohen_D
    df = df.join(_df)

    # Output to summary2.csv
    print("\tWriting " + output_file + ".")
    write_pd2csv(path, output_file, df, columnName, columnType, 1000)


def permutation(df, path, output):
    # Perform 1000 times permutation of relative timing between two mice during specified frames.
    # Compute overlapped freezing for each and statistical numbers

    import os
    import numpy as np
    import random

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    csStartFrame, csEndFrame, startFrame, endFrame = frames_range(
        video_system, video_total_frames)  # Get the frames range

    # Compute observed freezing, overlap, and the freezing bit maps (overlap)
    (_sub1Freeze, _sub2Freeze, _overlapFreeze, overlap, _fz_sub1_bout_dur_ave, _fz_sub1_bout_n, _fz_sub2_bout_dur_ave, _fz_sub2_bout_n) =\
        overlap_freezing3(df, path, False)

    # Permutation. Repeat random shift of subject1 for 1000 times
    nRepeat = 1000

    # Create working numpy array of % freezing time for each subject and % overlap
    sub1Freeze = np.array([0.0 for x in range(nRepeat)])
    sub2Freeze = np.array([0.0 for x in range(nRepeat)])
    overlapFreeze = np.array([0.0 for x in range(nRepeat)])

    for x in range(nRepeat):
        # Generate random number ranged from 0 to 479
        shift = random.randint(0, csEndFrame-csStartFrame)
        # Shift the freezing pattern in subject1 only during frame 241-720
        overlap[csStartFrame:csEndFrame, 0] = np.roll(
            overlap[csStartFrame:csEndFrame, 0], shift)

        # Scan the overlap variable for
        #   freezing in animal#1 (counter[0])
        #   freezing in animal#2 (counter[1])
        #   overlapped freezing (counter[2])
        counter = np.zeros((3), dtype=int)

        for i in range(csStartFrame, csEndFrame + 1):
            if overlap[i, 0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i, 1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i, 0] == 1 and overlap[i, 1] == 1:
                overlap[i, 2] = 1
                counter[2] = counter[2] + 1
            else:
                overlap[i, 2] = 0

        data_range = float(csEndFrame - csStartFrame+1)
        sub1Freeze[x] = counter[0]/data_range*100.0
        sub2Freeze[x] = counter[1]/data_range*100.0
        overlapFreeze[x] = counter[2]/data_range*100.0

    Cohen_D = (_overlapFreeze - np.mean(overlapFreeze)) / np.std(overlapFreeze)

    # Output to permutation.csv
    if output:
        print("\tWriting permutation.csv.")
        outputFilename = os.path.join(path, "_permutation.csv")
        print("\t\t" + outputFilename)
        output = open(outputFilename, "w")

        # Observed freezing and overlap
        output.write('sub1Freeze, sub2Freeze, overlapFreeze\n')
        output.write(
            str(_sub1Freeze) + ',' +
            str(_sub2Freeze) + ',' +
            str(_overlapFreeze) + '\n')

        # Permuted freezing and overlap
        for i in range(0, len(sub1Freeze)):
            output.write(
                str(sub1Freeze[i]) + ',' +
                str(sub2Freeze[i]) + ',' +
                str(overlapFreeze[i]) + ',' +
                'Permutation' + '\n')
        output.close()

        print("\tObserved overlap: {} \n\tTheoretical random overlap (mean): {} (SD): {} \n\tCohen_D: {}".format(
            _overlapFreeze, np.mean(overlapFreeze), np.std(overlapFreeze), Cohen_D))

    return(Cohen_D)


def permutation_org(df, path, output):
    # Perform 1000 times permutation of relative timing between two mice during tone.
    # Compute overlapped freeaing for each and statistical numbers

    # Compute overlap of freezing
    #
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    #     1 min for acclimation (frame 1-240: 240 frames)
    #     2 min for CS (frame 241-720: 480 frames))
    #
    #     Create np.array of 721 rows x 2 columes,
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     column 0 will be not used.
    # PCbox generates video 0-719 frames (total 720 frames)
    """
    Extended to generic epoch from home cage videos

    Parenthesis indicates the video frames to be compute for % epoch time
    FreezeFrame 0,  (pre: 1 - 240), (CS: 241 - 720), 721,   Total 722 frames
    PCBox           (pre: 0 - 239), (CS: 240 - 719),        Total 720 frames
    HomeCage - analyze the entire frames
                    (0 to video_total_frames - 1),          Total video_total_frames frames
    """

    import os
    import numpy as np
    import random

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    if video_system == 'FreezeFrame':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 241
        csEndFrame = 720
        column, row = 3, 721
    if video_system == 'FreezeFrame_1st_half':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 241
        csEndFrame = 480
        column, row = 3, 721
    if video_system == 'FreezeFrame_2nd_half':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 481
        csEndFrame = 720
        column, row = 3, 721
    if video_system == 'PCBox':
        # PCBox
        preStartFrame = 0
        preEndFrame = 239
        csStartFrame = 240
        csEndFrame = 719
        column, row = 3, 721
    if video_system == 'HomeCage':
        # Generic, presumably home cage annotation
        preStartFrame = 0
        preEndFrame = 0
        csStartFrame = 0
        csEndFrame = video_total_frames - 1
        column, row = 3, video_total_frames

    # Compute observed overlap
    # (_sub1Freeze, _sub2Freeze, _overlapFreeze,
    #  overlap) = overlap_freezing(df, path, False)
    (_sub1Freeze, _sub2Freeze, _overlapFreeze, overlap, fz_sub1_bout_dur_ave, fz_sub1_bout_n, fz_sub2_bout_dur_ave, fz_sub2_bout_n) =\
        overlap_freezing2(df, path, False)

    # Permutation. Repeat random shift of subject1 for 1000 times
    nRepeat = 1000

    # Create working numpy array of % freezing time for each subject and % overlap
    sub1Freeze = np.array([0.0 for x in range(nRepeat)])
    sub2Freeze = np.array([0.0 for x in range(nRepeat)])
    overlapFreeze = np.array([0.0 for x in range(nRepeat)])

    for x in range(nRepeat):
        # Generate random number ranged from 0 to 479
        shift = random.randint(0, csEndFrame-csStartFrame)
        # Shift the freezing pattern in subject1 only during frame 241-720
        overlap[csStartFrame:csEndFrame, 0] = np.roll(
            overlap[csStartFrame:csEndFrame, 0], shift)

        # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
        # and overlapped freezing (counter[2])
        counter = np.zeros((3), dtype=int)

        for i in range(csStartFrame, csEndFrame + 1):
            if overlap[i, 0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i, 1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i, 0] == 1 and overlap[i, 1] == 1:
                overlap[i, 2] = 1
                counter[2] = counter[2] + 1
            else:
                overlap[i, 2] = 0

        data_range = float(csEndFrame - csStartFrame+1)
        sub1Freeze[x] = counter[0]/data_range*100.0
        sub2Freeze[x] = counter[1]/data_range*100.0
        overlapFreeze[x] = counter[2]/data_range*100.0

    Cohen_D = (_overlapFreeze - np.mean(overlapFreeze)) / np.std(overlapFreeze)

    # Output to permutation.csv
    if output:
        print("\tWriting permutation.csv.")
        outputFilename = os.path.join(path, "_permutation.csv")
        print("\t\t" + outputFilename)
        output = open(outputFilename, "w")

        output.write('sub1Freeze, sub2Freeze, overlapFreeze\n')
        output.write(
            str(_sub1Freeze) + ',' +
            str(_sub2Freeze) + ',' +
            str(_overlapFreeze) + '\n')

        for i in range(0, len(sub1Freeze)):
            output.write(
                str(sub1Freeze[i]) + ',' +
                str(sub2Freeze[i]) + ',' +
                str(overlapFreeze[i]) + ',' +
                'Permutation' + '\n')
        output.close()

        print("\tObserved overlap: {} \n\tTheoretical random overlap (mean): {} (SD): {} \n\tCohen_D: {}".format(
            _overlapFreeze, np.mean(overlapFreeze), np.std(overlapFreeze), Cohen_D))

    return(Cohen_D)


def p_compute_cohen_d(path, columnName, columnType, start, end, input_file='summary1.csv'):
    # Read CSV into pandas DF
    df = read_csv2pd(path, input_file, columnName, columnType)

    # Compute permutation/Cohen_D and store in DF
    print("\tProcessing column: ", end=" ")
    Cohen_D = np.zeros(len(df))

    if end > len(df):
        end = len(df)
    if start >= len(df):
        return

    for i in range(start, end):
        print("{}/{}, ".format(i, len(df)), end=" ")
        # if i % 1000 == 0:
        #     print("{}/{}, ".format(i, len(df)), end=" ")
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            # print("hit!")
            Cohen_D[i] = "nan"
        else:
            subfolder = os.path.join(path, df.iloc[i, 0])
            Cohen_D[i] = permutation(df.iloc[i, :], subfolder, False)
    print("completed.")

    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[-1]] = Cohen_D
    df = df.join(_df)

    # Output to summary2.csv
    # print("\tWriting " + output_file + ".")
    # write_pd2csv(path, output_file, df, columnName, columnType, 1000)

    return df


def p_permutation(df, path, output, id):
    # Perform 1000 times permutation of relative timing between two mice during tone.
    # Compute overlapped freeaing for each and statistical numbers

    # Compute overlap of freezing
    #
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    #     1 min for acclimation (frame 1-240: 240 frames)
    #     2 min for CS (frame 241-720: 480 frames))
    #
    #     Create np.array of 721 rows x 2 columes,
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     column 0 will be not used.
    # PCbox generates video 0-719 frames (total 720 frames)
    """
    Extended to generic epoch from home cage videos

    Parenthesis indicates the video frames to be compute for % epoch time
    FreezeFrame 0,  (pre: 1 - 240), (CS: 241 - 720), 721,   Total 722 frames
    PCBox           (pre: 0 - 239), (CS: 240 - 719),        Total 720 frames
    HomeCage - analyze the entire frames
                    (0 to video_total_frames - 1),          Total video_total_frames frames
    """

    import os
    import numpy as np
    import random

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    if video_system == 'FreezeFrame':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 241
        csEndFrame = 720
        column, row = 3, 721
    if video_system == 'PCBox':
        # PCBox
        preStartFrame = 0
        preEndFrame = 239
        csStartFrame = 240
        csEndFrame = 719
        column, row = 3, 721
    if video_system == 'HomeCage':
        # Generic, presumably home cage annotation
        preStartFrame = 0
        preEndFrame = 0
        csStartFrame = 0
        csEndFrame = video_total_frames - 1
        column, row = 3, video_total_frames

    # Compute observed overlap
    (_sub1Freeze, _sub2Freeze, _overlapFreeze,
     overlap) = overlap_freezing(df, path, False)

    # Permutation. Repeat random shift of subject1 for 1000 times
    nRepeat = 1000

    # Create working numpy array of % freezing time for each subject and % overlap
    sub1Freeze = np.array([0.0 for x in range(nRepeat)])
    sub2Freeze = np.array([0.0 for x in range(nRepeat)])
    overlapFreeze = np.array([0.0 for x in range(nRepeat)])

    for x in range(nRepeat):
        # Generate random number ranged from 0 to 479
        shift = random.randint(0, csEndFrame-csStartFrame)
        # Shift the freezing pattern in subject1 only during frame 241-720
        overlap[csStartFrame:csEndFrame, 0] = np.roll(
            overlap[csStartFrame:csEndFrame, 0], shift)

        # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
        # and overlapped freezing (counter[2])
        counter = np.zeros((3), dtype=int)

        for i in range(csStartFrame, csEndFrame + 1):
            if overlap[i, 0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i, 1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i, 0] == 1 and overlap[i, 1] == 1:
                overlap[i, 2] = 1
                counter[2] = counter[2] + 1
            else:
                overlap[i, 2] = 0

        data_range = float(csEndFrame - preEndFrame)
        sub1Freeze[x] = counter[0]/data_range*100.0
        sub2Freeze[x] = counter[1]/data_range*100.0
        overlapFreeze[x] = counter[2]/data_range*100.0

    Cohen_D = (_overlapFreeze - np.mean(overlapFreeze)) / np.std(overlapFreeze)

    # Output to permutation.csv
    if output:
        print("\tWriting permutation.csv.")
        outputFilename = os.path.join(path, "_permutation.csv")
        print("\t\t" + outputFilename)
        output = open(outputFilename, "w")

        output.write('sub1Freeze, sub2Freeze, overlapFreeze\n')
        output.write(
            str(_sub1Freeze) + ',' +
            str(_sub2Freeze) + ',' +
            str(_overlapFreeze) + '\n')

        for i in range(0, len(sub1Freeze)):
            output.write(
                str(sub1Freeze[i]) + ',' +
                str(sub2Freeze[i]) + ',' +
                str(overlapFreeze[i]) + ',' +
                'Permutation' + '\n')
        output.close()

        print("\tObserved overlap: {} \n\tTheoretical random overlap (mean): {} (SD): {} \n\tCohen_D: {}".format(
            _overlapFreeze, np.mean(overlapFreeze), np.std(overlapFreeze), Cohen_D))

    return [id, Cohen_D]

    # Perform 1000 times permutation of relative timing between two mice during tone.
    # Compute overlapped freeaing for each and statistical numbers

    # Compute overlap of freezing
    #
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    #     1 min for acclimation (frame 1-240: 240 frames)
    #     2 min for CS (frame 241-720: 480 frames))
    #
    #     Create np.array of 721 rows x 2 columes,
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     column 0 will be not used.
    import os
    import numpy as np
    import random

    # Compute overlap table
    (_sub1Freeze, _sub2Freeze, _overlapFreeze,
     overlap) = overlap_freezing(df, video_system, path, False)

    # Permutation
    # Repeat random shift of subject1 for 1000 times
    nRepeat = 1000

    sub1Freeze = np.array([0.0 for x in range(nRepeat)])
    sub2Freeze = np.array([0.0 for x in range(nRepeat)])
    overlapFreeze = np.array([0.0 for x in range(nRepeat)])

    for x in range(nRepeat):
        # Generate random number ranged from 0 to 479
        shift = random.randint(0, 479)
        # Shift the freezing pattern in subject1 only during frame 241-720
        overlap[241:720, 0] = np.roll(overlap[241:720, 0], shift)

        # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
        # and overlapped freezing (counter[2])
        counter = np.zeros((3), dtype=int)

        for i in range(241, len(overlap)):
            if overlap[i, 0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i, 1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i, 0] == 1 and overlap[i, 1] == 1:
                overlap[i, 2] = 1
                counter[2] = counter[2] + 1
            else:
                overlap[i, 2] = 0

        sub1Freeze[x] = counter[0]/480.0*100.0
        sub2Freeze[x] = counter[1]/480.0*100.0
        overlapFreeze[x] = counter[2]/480.0*100.0

    Cohen_D = (_overlapFreeze - np.mean(overlapFreeze)) / np.std(overlapFreeze)

    # Output to permutation.csv
    if output:
        print("\tWriting permutation.csv.")
        outputFilename = os.path.join(path, "_permutation.csv")
        print("\t\t" + outputFilename)
        output = open(outputFilename, "w")

        output.write('sub1Freeze, sub2Freeze, overlapFreeze\n')
        output.write(
            str(_sub1Freeze) + ',' +
            str(_sub2Freeze) + ',' +
            str(_overlapFreeze) + '\n')

        for i in range(0, len(sub1Freeze)):
            output.write(
                str(sub1Freeze[i]) + ',' +
                str(sub2Freeze[i]) + ',' +
                str(overlapFreeze[i]) + ',' +
                'Permutation' + '\n')
        output.close()

        print("\tObserved overlap: {} \n\tTheoretical random overlap (mean): {} (SD): {} \n\tCohen_D: {}".format(
            _overlapFreeze, np.mean(overlapFreeze), np.std(overlapFreeze), Cohen_D))

    return(Cohen_D)


###############################################################################
# compute_lagtime()
#       lag_time()
#           lagtime()
#       write_pd2csv()
def compute_lagtime():
    # Read CSV into pandas DF
    df = read_csv2pd(path, 'summary2.csv', columnName, columnType)

    # Compute permutation/Cohen_D and store in DF
    print("\nStep4. Computing lag times.")

    columnName = np.append(columnName, [
                           'lagt_start_s1_s2', 'lagt_start_s2_s1', 'lagt_end_s1_s2', 'lagt_end_s2_s1'])
    columnType = np.append(
        columnType, ['int_array', 'int_array', 'int_array', 'int_array'])

    s1_s2_start = np.empty((len(df),), dtype=object)
    s2_s1_start = np.empty((len(df),), dtype=object)
    s1_s2_end = np.empty((len(df),), dtype=object)
    s2_s1_end = np.empty((len(df),), dtype=object)

    for i in range(0, len(df)):
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            # print("hit!")
            s1_s2_start[i], s2_s1_start[i], s1_s2_end[i], s2_s1_end[i] = np.array(
                []), np.array([]), np.array([]), np.array([])
        else:
            s1_s2_start[i], s2_s1_start[i], s1_s2_end[i], s2_s1_end[i] = lag_time(
                df.iloc[i, :], DEBUG=False)

    # Add columns & data
    _df = pd.DataFrame()
    i = 12
    _df[columnName[i]] = s1_s2_start
    _df[columnName[i+1]] = s2_s1_start
    _df[columnName[i+2]] = s1_s2_end
    _df[columnName[i+3]] = s2_s1_end
    df = df.join(_df)

    # Output to summary3.csv
    print("\tWriting summary3.csv.")
    write_pd2csv(path, 'summary3.csv', df, columnName, columnType, 1000)


def lag_time(df, DEBUG=False):

    s1 = df['fz_start_sub1']
    s2 = df['fz_start_sub2']
    s1_s2_start = lagtime(s1, s2, DEBUG)  # Freezing onset from s1 mouse to s2

    s1 = df['fz_start_sub2']
    s2 = df['fz_start_sub1']
    s2_s1_start = lagtime(s1, s2, DEBUG)  # Freezing onset from s1 mouse to s2

    s1 = df['fz_end_sub1']
    s2 = df['fz_end_sub2']
    s1_s2_end = lagtime(s1, s2, DEBUG)  # Freezing onset from s1 mouse to s2

    s1 = df['fz_end_sub2']
    s2 = df['fz_end_sub1']
    s2_s1_end = lagtime(s1, s2, DEBUG)  # Freezing onset from s1 mouse to s2

    return(s1_s2_start, s2_s1_start, s1_s2_end, s2_s1_end)


def lagtime(w1, w2, DEBUG=False):
    # Search the closest epochs from partner (w2)
    # lag time > 0 means w2 follows w1

    import numpy as np

    # Indices of the closest freezing epoch from partner
    indexCloseset = np.zeros(len(w1), dtype=int)
    lagTime = np.zeros(len(w1), dtype=int)   # lag time (frame number)

    for i in range(0, len(w1)):
        _lagTime = 10000
        _indexCloseset = 0
        # Serach in w2
        for j in range(0, len(w2)):
            if abs(_lagTime) > abs(w2[j] - w1[i]):
                _indexCloseset = j
                _lagTime = w2[j] - w1[i]

        indexCloseset[i] = _indexCloseset
        lagTime[i] = _lagTime

    if DEBUG:
        print("Sub1 epoch ID     :", end=" ")
        print(*range(0, len(w1)), sep=", ")

        print("Sub1 frame number :", end=" ")
        print(*w1, sep=", ")

        print("Sub2 freeze epoch :", end=" ")
        print(*indexCloseset, sep=", ")

        print("Sub2 frame number :", end=" ")
        print(*w2[indexCloseset], sep=", ")

        print("lag-time          :", end=" ")
        print(*lagTime, sep=", ")

    return(lagTime)


###############################################################################
# compute_markov_chain()
#       read_csv2pd()
#       state_trans()
#       write_pd2csv()
def compute_markov_chain(path, columnName, columnType, input_file='summary3.csv', output_file='summary4.csv'):
    # Read CSV into pandas DF
    df = read_csv2pd(path, input_file, columnName, columnType)

    # Count behavioral state transitions for Markov chain analysis
    print("\tProcessing column: ", end=" ")

    # Initalize the parameters
    state_count = np.zeros((len(df), 4), dtype=int)
    state_trans_count = np.zeros((len(df), 16), dtype=int)

    # Initialize the state_change table
    state_trans_tp = pd.DataFrame()
    state_trans_tp['ID'] = []
    state_trans_tp['From'] = []
    state_trans_tp['To'] = []
    state_trans_tp['frame_n'] = []
    state_trans_tp = state_trans_tp.astype(int)

    # Main loop
    for i in range(0, len(df)):
        if i % 10 == 0:
            print("{}/{}, ".format(i, len(df)), end=" ")

        # If single animal, do nothing.
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            pass
        else:
            state_count[i, :], state_trans_count[i, :], state_trans_tp = \
                state_trans(df.iloc[i, :], df['video_system']
                            [i], state_trans_tp, DEBUG=False)

    print("completed.")

    # Add columns & data
    _df = pd.DataFrame()
    for i in range(4):
        _df[columnName[-(4+16-i)]] = state_count[:, i]
        # _df[columnName[i+15]] = state_count[:, i]
    for i in range(16):
        _df[columnName[-(16-i)]] = state_trans_count[:, i]
        # _df[columnName[i+19]] = state_trans_count[:, i]
    df = df.join(_df)

    # Output to summary3.csv
    print("\tWriting " + output_file + ".")
    write_pd2csv(path, output_file, df, columnName, columnType, 1000)

    print("\tWriting " + output_file[0:-4] + "_state_trans.csv.")
    columnName = ['ID',   'From',    'To',     'frame_n']
    columnType = ['str',  'int',     'int',    'int']
    write_pd2csv(path, output_file[0:-4] + '_state_trans.csv', state_trans_tp,
                 columnName, columnType, 1000)


def state_trans(df, video_system, state_trans_tp, DEBUG=False):
    import numpy as np
    import pandas as pd

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    csStartFrame, csEndFrame, startFrame, endFrame = frames_range(
        video_system, video_total_frames)  # Get the frames range

    #################################################
    # Generate pandas dataframe for freezing states
    #    (bools) 'sub1_freeze', 'sub2_freeze', 'other' for each video frame
    #################################################
    # Create working numpy matrix of freezing (True/False) for the pandas dataframe
    # All values are False
    sub_freeze = np.zeros((endFrame + 1, 3), dtype=bool)

    # For animal#1
    for i in range(0, int(len(sub1Start))):
        for j in range(sub1Start[i], sub1End[i] + 1):
            if j >= csStartFrame and j <= csEndFrame:
                sub_freeze[j][0] = True

    # For animal#2
    for i in range(0, int(len(sub2Start))):
        for j in range(sub2Start[i], sub2End[i] + 1):
            if j >= csStartFrame and j <= csEndFrame:
                sub_freeze[j][1] = True

    # Initialize the pandas DataFrame
    _df = pd.DataFrame(data=sub_freeze, columns=[
                       'sub1_freeze', 'sub2_freeze', 'other'], dtype=bool)
    #################################################
    # Identify and count each behavior states
    #     state  sub1    sub2
    #     0      freeze  freeze
    #     1      freeze  no
    #     2      no      freeze
    #     3      no      no
    #################################################
    state = np.zeros(len(_df), dtype=int)
    state_count = np.zeros(4, dtype=int)

    for i in range(0, len(_df)):
        if _df['sub1_freeze'][i] == True and _df['sub2_freeze'][i] == True:
            state[i] = 0
            if i >= csStartFrame and i <= csEndFrame:
                state_count[0] += 1
        elif _df['sub1_freeze'][i] == True and _df['sub2_freeze'][i] == False:
            state[i] = 1
            if i >= csStartFrame and i <= csEndFrame:
                state_count[1] += 1
        elif _df['sub1_freeze'][i] == False and _df['sub2_freeze'][i] == True:
            state[i] = 2
            if i >= csStartFrame and i <= csEndFrame:
                state_count[2] += 1
        elif _df['sub1_freeze'][i] == False and _df['sub2_freeze'][i] == False:
            state[i] = 3
            if i >= csStartFrame and i <= csEndFrame:
                state_count[3] += 1

    # merge the states into the _df as state column
    # _df: (bools) 'sub1_freeze', 'sub2_freeze', 'other', (int) state
    # store the results to _df2 temporarily
    _df2 = pd.DataFrame()
    _df2['state'] = state
    _df = _df.join(_df2)

    #################################################
    # Create dataframe of the From state and To state
    #    Because it's transition, the total number of row decreases by 1
    #################################################
    _df2 = pd.DataFrame()
    _df2['From'] = []
    _df2['To'] = []
    _df2 = _df2.astype(int)

    # Because it's transition, the total number of row decreases by 1
    for i in range(0, len(_df)-1):
        new_row = {'From': _df['state'][i], 'To': _df['state'][i+1]}
        # Need to wrap in a list to avoid an error
        new_row = pd.DataFrame([new_row])
        # 20220331 wi: switch from pd.append to pd.concat
        _df2 = pd.concat([_df2, new_row], ignore_index=True)

    #################################################
    # Using the _df2, identify and count each transition from the 16 patterns
    #   during CS (either 241-720 or 240-719). Frame number
    #   is 480, but the number of states is 479)
    #################################################
    pattern = [[0, 0], [0, 1], [0, 2], [0, 3],
               [1, 0], [1, 1], [1, 2], [1, 3],
               [2, 0], [2, 1], [2, 2], [2, 3],
               [3, 0], [3, 1], [3, 2], [3, 3]]
    state_trans_count = np.zeros(16, dtype=int)

    for i in range(csStartFrame, csEndFrame):
        for j in range(len(pattern)):
            if _df2['From'][i] == pattern[j][0] and _df2['To'][i] == pattern[j][1]:
                state_trans_count[j] += 1

    #################################################
    # Concatenate dataframe of ID, From, To, frame_n
    #################################################
    for i in range(csStartFrame, csEndFrame):
        if _df2['From'][i] != _df2['To'][i]:
            new_row = {'ID': df['folder_videoname'],
                       'From': _df2['From'][i], 'To': _df2['To'][i], 'frame_n': i}
            new_row = pd.DataFrame([new_row])
            # state_trans_tp = state_trans_tp.append(new_row, ignore_index=True)
            state_trans_tp = pd.concat(
                [state_trans_tp, new_row], ignore_index=True)

    return(state_count, state_trans_count, state_trans_tp)


def state_trans_org(df, video_system, state_trans_tp, DEBUG=False):
    import numpy as np
    import pandas as pd

    video_system = df['video_system']
    video_total_frames = df['video_total_frames']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    """
    Video frame to be analyzed
    FreezeFrame 0, (pre: 1 - 240), (CS: 241 - 720), 721, Total 722 frames
    PCBox          (pre: 0 - 239), (CS: 240 - 719),      Total 720 frames
    """

    if video_system == 'FreezeFrame':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 241
        csEndFrame = 720
        column, row = 3, 721
    if video_system == 'PCBox':
        # PCBox
        preStartFrame = 0
        preEndFrame = 239
        csStartFrame = 240
        csEndFrame = 719
        column, row = 3, 721
    if video_system == 'HomeCage':
        # Generic, presumably home cage annotation
        preStartFrame = 0
        preEndFrame = 0
        csStartFrame = 0
        csEndFrame = video_total_frames - 1
        column, row = 3, video_total_frames
    # Create working numpy matrix of freezing (True/False) for each video frame
    #column, row = 3, 721

    sub_freeze = np.zeros((row, column), dtype=bool)  # All values are False

    # Mark True when freeze
    # For animal#1
    for i in range(0, int(len(sub1Start))):
        if sub1Start[i] < preStartFrame:
            sub1Start[i] = preStartFrame
        if sub1End[i] > csEndFrame:
            sub1End[i] = csEndFrame
        for j in range(sub1Start[i], sub1End[i]+1):
            sub_freeze[j][0] = True

    # For animal#2
    for i in range(0, int(len(sub2Start))):
        if sub2Start[i] < preStartFrame:
            sub2Start[i] = preStartFrame
        if sub2End[i] > csEndFrame:
            sub2End[i] = csEndFrame
        for j in range(sub2Start[i], sub2End[i]+1):
            sub_freeze[j][1] = True

    # Initialize Pandas DataFrame as df. The size is the same with sub_freeze (frame 0-720)
    _df = pd.DataFrame(data=sub_freeze, columns=[
                       'sub1_freeze', 'sub2_freeze', 'other'], dtype=bool)
    #################################################
    # compute the behavior states
    #     state  sub1    sub2
    #     0      freeze  freeze
    #     1      freeze  no
    #     2      no      freeze
    #     3      no      no
    #################################################
    state = np.zeros(len(_df), dtype=int)
    state_count = np.zeros(4, dtype=int)

    for i in range(0, len(_df)):
        if _df['sub1_freeze'][i] == True and _df['sub2_freeze'][i] == True:
            state[i] = 0
            if i >= csStartFrame and i <= csEndFrame:
                state_count[0] += 1
        elif _df['sub1_freeze'][i] == True and _df['sub2_freeze'][i] == False:
            state[i] = 1
            if i >= csStartFrame and i <= csEndFrame:
                state_count[1] += 1
        elif _df['sub1_freeze'][i] == False and _df['sub2_freeze'][i] == True:
            state[i] = 2
            if i >= csStartFrame and i <= csEndFrame:
                state_count[2] += 1
        elif _df['sub1_freeze'][i] == False and _df['sub2_freeze'][i] == False:
            state[i] = 3
            if i >= csStartFrame and i <= csEndFrame:
                state_count[3] += 1

    # store the results to _df2 temporarily
    _df2 = pd.DataFrame()
    _df2['state'] = state
    # merge the result as new column in _df
    _df = _df.join(_df2)
    #################################################
    # compute the state transition
    #    Number of row decreases by 1 (0-719)
    #################################################
    _df2 = pd.DataFrame()
    _df2['From'] = []
    _df2['To'] = []
    _df2 = _df2.astype(int)

    for i in range(0, len(_df)-1):
        new_row = {'From': _df['state'][i], 'To': _df['state'][i+1]}
        # Need to wrap in a list to avoid an error
        new_row = pd.DataFrame([new_row])
        # 20220331 wi: switch from pd.append to pd.concat
        _df2 = pd.concat([_df2, new_row], ignore_index=True)

    #################################################
    # Search for each of 16 patterns
    #   during CS (either 241-720 or 240-719). Frame number
    #   is 480, but the number of states is 479)
    #################################################
    pattern = [[0, 0], [0, 1], [0, 2], [0, 3],
               [1, 0], [1, 1], [1, 2], [1, 3],
               [2, 0], [2, 1], [2, 2], [2, 3],
               [3, 0], [3, 1], [3, 2], [3, 3]]
    state_trans_count = np.zeros(16, dtype=int)

    for i in range(csStartFrame, csEndFrame):
        for j in range(len(pattern)):
            if _df2['From'][i] == pattern[j][0] and _df2['To'][i] == pattern[j][1]:
                state_trans_count[j] += 1

    # print(state_trans_count)

    #################################################
    # Search only transition (12 patterns), and extract timepoint
    #################################################
    # Initialize table (1000 rows, 3 columns)
    # state_trans_tp = pd.DataFrame()
    # state_trans_tp['ID'] = []
    # state_trans_tp['From'] = []
    # state_trans_tp['To'] = []
    # state_trans_tp['frame_n'] = []
    # state_trans_tp = state_trans_tp.astype(int)

    # state_trans_tp = np.zeros((1000, 3), dtype=int)
    # c_trans = 0
    for i in range(csStartFrame, csEndFrame):
        if _df2['From'][i] != _df2['To'][i]:
            new_row = {'ID': df['folder_videoname'],
                       'From': _df2['From'][i], 'To': _df2['To'][i], 'frame_n': i}
            new_row = pd.DataFrame([new_row])
            # state_trans_tp = state_trans_tp.append(new_row, ignore_index=True)
            state_trans_tp = pd.concat(
                [state_trans_tp, new_row], ignore_index=True)
            # _df2 = pd.concat([_df2, new_row], ignore_index=True)
    # for i in range(csStartFrame, csEndFrame):
    #     if _df2['From'][i] != _df2['To'][i]:
    #         state_trans_tp[c_trans][0] = _df2['From'][i]
    #         state_trans_tp[c_trans][1] = _df2['To'][i]
    #         state_trans_tp[c_trans][2] = i
    #         c_trans += 1

    return(state_count, state_trans_count, state_trans_tp)

###############################################################################
# compute_distance()
#       read_csv2pd()
#       distance_cs()
#       write_pd2csv()


def compute_distance():
    # Read CSV into pandas DF
    df = read_csv2pd(path, 'summary4.csv', columnName, columnType)
    df_traj = read_csv2pd(path, 'summary_traj.csv',
                          columnName_traj, columnType_traj)
    id_list = df_traj.id.unique()

    # Compute averaged distance during CS and store in DF
    print("\nStep6. Computing averaged distance during CS.")

    columnName = np.append(columnName, ['dis_cs'])
    columnType = np.append(columnType, ['float'])

    dis_cs = np.zeros(len(df))

    for i in range(0, len(df)):
        if np.where(id_list == df['folder_videoname'][i])[0].size:
            dis_cs[i] = distance_cs(
                df.iloc[i, :], df_traj[df_traj["id"] == df['folder_videoname'][i]], df['video_system'][i])
        else:
            dis_cs[i] = "nan"

    # Add columns & data
    _df = pd.DataFrame()
    i = 35
    _df[columnName[i]] = dis_cs
    df = df.join(_df)

    # Output to summary1.csv
    print("\tWriting summary5.csv.")
    write_pd2csv(path, 'summary5.csv', df, columnName, columnType, 1000)


def distance_cs(df, df_traj, video_system):

    # Compute overlap of freezing
    #
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    # 1) FreezeFrame generates video 0-721 frames (total 722 frames)
    # Only care 1-720 frames (total 720 frames), ignoring frame# 0 (black frame) and 721.
    #     1 min for acclimation (frame 1-240)
    #     2 min for CS (frame 241-720))
    #
    #     Create np.array of 721 rows x 2 columes,
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     row 0 will be not used.

    # PCbox generates video 0-719 frames (total 720 frames)

    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import math

    video_system = df['video_system']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']

    """
    Video frame to be analyzed
    FreezeFrame 0, (pre: 1 - 240), (CS: 241 - 720), 721, Total 722 frames
    PCBox          (pre: 0 - 239), (CS: 240 - 719),      Total 720 frames
    """

    if video_system == 'FreezeFrame':
        # FreezeFrame
        preStartFrame = 1
        preEndFrame = 240
        csStartFrame = 241
        csEndFrame = 720
    if video_system == 'PCBox':
        # PCBox
        preStartFrame = 0
        preEndFrame = 239
        csStartFrame = 240
        csEndFrame = 719

    sub1x = df_traj[0:1][list(
        map(str, list(range(csStartFrame, csEndFrame + 1))))].to_numpy()[0]
    sub1y = df_traj[1:2][list(
        map(str, list(range(csStartFrame, csEndFrame + 1))))].to_numpy()[0]
    sub2x = df_traj[2:3][list(
        map(str, list(range(csStartFrame, csEndFrame + 1))))].to_numpy()[0]
    sub2y = df_traj[3:4][list(
        map(str, list(range(csStartFrame, csEndFrame + 1))))].to_numpy()[0]

    print('sub1x', sub1x, sub1x.size)
    print('sub1y', sub1y, sub1y.size)
    print('sub2x', sub2x, sub2x.size)
    print('sub2y', sub2y, sub2y.size)

    dis = 0.0

    for i in range(csEndFrame - csStartFrame + 1):
        dis = dis + math.sqrt((sub1x[i] - sub2x[i])
                              ** 2 + (sub1y[i] - sub2y[i])**2)

    dis = dis / (csEndFrame - csStartFrame + 1)

    return(dis)

###############################################################################


sys.path.insert(1, r'W:\wataru\jupyter\synchro_freeze')


def comp_markov_cumm_prob(a):
    #################################
    # 4 states Markov chain simulation
    # Input:
    # a: Sampled counts of each transition
    # <Example>
    #    [76,3, 12,0, 4, 48,1, 5, 10,0, 154,15,1, 7, 12,131]
    #     00,01,02,03,10,11,12,13,20,21,22, 23,30,31,32,33
    # frame_n: How many step compute the simulation
    #
    ##############
    # Compute transition probabilities from the sampled counts
    a_prob = np.zeros((4, 4))

    # Compute each probability
    for i in range(4):
        _sum = 0
        for j in range(4):
            _sum = _sum + a[i*4 + j]
        for j in range(4):
            # In some case, all counts are zero.
            if _sum == 0:
                print('all transition probabilities are zero')
                a_prob[i][j] = 0
            else:
                a_prob[i][j] = float(a[i*4 + j]) / float(_sum)
    # print(a_prob)

    # Compute cummurative probability
    for i in range(4):
        for j in range(1, 4):
            a_prob[i][j] = a_prob[i][j] + a_prob[i][j-1]
    # print(a_prob)
    return a_prob


def comp_markov(a_prob, frame_n):
    ##############
    # Markov chain simulation
    # Initial state is both are no-freeze (state:3)
    state_id = 3

    state_seq = np.zeros(frame_n)

    for i in range(frame_n):
        # two choice for generate random number
        _random = random.SystemRandom().uniform(0, 1)
        #_random = random.uniform(0, 1)

        # take transition probabilities in a specific state
        prob = np.append([0.], a_prob[state_id])
        # print(_random, prob)

        for j in range(4):
            if prob[j] <= _random and _random < prob[j+1]:
                # DEBUG
                # print(i, _random, state_id, j)
                state_seq[i] = j
                state_id = j
                break

    return state_seq


def comp_freeze_pattern(state_seq, frame_n):
    ##############
    # Mark freeze as 1 in the freeze_pattern
    column, row = 2, frame_n
    freeze_pattern = np.array([[0 for x in range(column)] for y in range(row)])

    for i in range(len(state_seq)):
        if state_seq[i] == 0:
            freeze_pattern[i][0], freeze_pattern[i][1] = 1, 1
        if state_seq[i] == 1:
            freeze_pattern[i][0], freeze_pattern[i][1] = 1, 0
        if state_seq[i] == 2:
            freeze_pattern[i][0], freeze_pattern[i][1] = 0, 1
        if state_seq[i] == 3:
            freeze_pattern[i][0], freeze_pattern[i][1] = 0, 0

    return freeze_pattern


def comp_freeze_bout(sub, freeze_pattern):
    #################################
    # Extract freeze state for each subject
    # Compute subStart, subEnd
    # Reject as a freeze bout if < 4

    _size = 1000
    freeze_state = -1
    freeze_count = -1
    subStart = np.full(_size, -1, dtype=int)
    subEnd = np.full(_size, -1, dtype=int)

    for i in range(len(freeze_pattern)):
        if freeze_pattern[i][sub] != freeze_state:
            freeze_state = freeze_pattern[i][sub]
            if freeze_state == 1:
                freeze_count += 1
                subStart[freeze_count] = i
            elif freeze_count != -1:  # subStart entry should exist
                if i-subStart[freeze_count] >= 3:
                    subEnd[freeze_count] = i
                else:  # Wind up
                    subStart[freeze_count] = -1
                    freeze_count -= 1
    # At the end of the for loop, if the for loop ends with freeze, set subEnd with the last frame
    if freeze_state == 1:
        subEnd[freeze_count] = len(freeze_pattern)-1

    return subStart[subStart != -1], subEnd[subEnd != -1]


def plot_freeze(freeze_pattern):
    import matplotlib.pyplot as plt

    # Plotting the freezing dynamics
    fig = plt.figure(num=None, figsize=(15, 2), dpi=80,
                     facecolor='w', edgecolor='k')
    # fig.subplots_adjust(top=0.8)

    ax1 = fig.add_subplot(111)
    x = freeze_pattern[:, 0] + 1.75
    y = freeze_pattern[:, 1] + 1.25

    ax1.plot(x)
    ax1.plot(y)

    ax1.set_ylabel('freeze or no freeze')
    ax1.set_xlabel('frame number')
    ax1.set_title('From top: Animal1, Animal2')

    plt.show()


###############################################################################
# mk_create_master_table()
#       _read_csv()
#       write_pd2csv()
#           preprocess_output_str()
def mk_create_master_table(path, columnName, columnType, sampled_counts, sample_n, frame_n,
                           preamble, postamble, output_file='summary1.csv'):

    # Initialize Pandas DataFrame for master table
    df = pd.DataFrame()
    for i in range(len(columnName)):
        df[columnName[i]] = []

    print("\tProcessing:  ", end=" ")

    #################################################################
    # Markov simulation
    #   Sampled counts of each transition
    # sampled_counts = np.array([76,3,12,0,4,48,1,5,10,0,154,15,1,7,12,131])
    # How many cycles of discreate time
    # frame_n = 480

    for j in range(len(sampled_counts)):
        print(j, end=' ')
        # extract counts of each transition
        sampled_count = sampled_counts.loc[[j], ['st_count_00', 'st_count_01', 'st_count_02', 'st_count_03',
                                                 'st_count_10', 'st_count_11', 'st_count_12', 'st_count_13',
                                                 'st_count_20', 'st_count_21', 'st_count_22', 'st_count_23',
                                                 'st_count_30', 'st_count_31', 'st_count_32', 'st_count_33', ]].to_numpy()[0]
        cumm_prob = comp_markov_cumm_prob(sampled_count)

        # extract group name
        mouse_id = sampled_counts.iloc[j]['folder_videoname']

        for i in range(sample_n):
            state_seq = comp_markov(cumm_prob, frame_n)

            # Compute freeze pattern for each mouse
            freeze_pattern = comp_freeze_pattern(state_seq, frame_n)

            # Generate freeze pattern in FreezeFrame format
            #   Add [0,0] * 241 at the bigining
            #   Add [0,0] * 1   at the end
            """
            Extended to generic epoch from home cage videos

            Parenthesis indicates the video frames to be compute for % epoch time
            FreezeFrame 0,  (pre: 1 - 240), (CS: 241 - 720), 721,   Total 722 frames
            PCBox           (pre: 0 - 239), (CS: 240 - 719),        Total 720 frames
            HomeCage - analyze the entire frames
                            (0 to video_total_frames - 1),          Total video_total_frames frames
            """
            freeze_pattern = np.concatenate(
                (np.full((preamble, 2), 0), freeze_pattern, np.full((postamble, 2), 0)))

            # Plot freeze pattern
            # plot_freeze(freeze_pattern)

            # Compute arrays for frame_number for freeze_start and end
            sub1Start, sub1End = comp_freeze_bout(0, freeze_pattern)
            sub2Start, sub2End = comp_freeze_bout(1, freeze_pattern)

            # Concatenate the freeze pattern to the df
            _df = {
                columnName[0]: mouse_id + '_' + str(i),
                columnName[1]: 'FALSE',
                columnName[2]: 'FreezeFrame',
                columnName[3]: 721,
                columnName[4]: [sub1Start],
                columnName[5]: [sub1End],
                columnName[6]: [sub2Start],
                columnName[7]: [sub2End]
            }
            _df = pd.DataFrame.from_dict(_df)
            df = pd.concat([df, _df], ignore_index=True)

    # Output to summary.csv
    print("\tWriting" + output_file + ".")
    sf.write_pd2csv(path, output_file, df, columnName, columnType, 1000)


# def write_pd2csv(path, filename, df, columnName, columnType, mlw=1000):
#     import os
#     import numpy as np
#     import pandas as pd

#     outputFilename = os.path.join(path, filename)
#     output = open(outputFilename, "w")
#     # mlw = 1000 # max_line_width in np.array2string

#     output.write(','.join(columnName)+'\n')

#     for i in range(0, len(df)):
#         output_str = ''
#         for j in range(0, len(columnName)):
#             # print(df.shape,j)
#             output_str = preprocess_output_str(
#                 output_str, df.iloc[i, j], columnType[j], 1000)
#         output.write(output_str[0:-1] + '\n')
#     output.close()
#     return


# def preprocess_output_str(output_str, data, columnType, mlw=1000):
#     import numpy as np

#     if columnType == 'int_array':
#         output_str = output_str + \
#             np.array2string(data, max_line_width=mlw) + ','
#     elif columnType == 'float' or columnType == 'bool' or columnType == 'int':
#         output_str = output_str + str(data) + ','
#     elif columnType == 'str':
#         output_str = output_str + data + ','

#     return(output_str)
