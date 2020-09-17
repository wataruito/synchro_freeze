# extracting onset and offset of freezing from two subject from a csv file
'''
File format for the csv file:
Currently frame number and sec appear together. Need to clean up.

dir:			20191015
old_dir:		101519
exp_id:			m38
single_animal:	TRUE (optional)
comment:		m38	muscimole, SINGLE

data:	start	end		duration	start	end		duration
		256		271		4			256		340		21.25
		273		276		1			389		394		1.5
		278		340		15.75		410		424		3.75
		365		385		5.25		431		448		4.5
		455		459		1.25		452		459		2
		465		491		6.75		465		491		6.75
		509		518		2.5			493		505		3.25
		531		549		4.75		523		533		2.75
		552		558		1.75		543		558		4
		560		564		1.25		563		582		5
		567		589		5.75		586		589		1
		597		603		1.75		592		599		2
		607		637		7.75		603		619		4.25
		651		660		2.5			623		635		3.25
		664		688		6.25		642		651		2.5
		699		711		3.25		654		661		2
									665		673		2.25
									677		696		5
									701		708		2

comment:	total (s)	71.5						79
			%			59.58333333					65.83333333
end:
'''

##################################################################################################
def process_freeze(path, DEBUG):
    
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd

    # DEBUG = 0
    #############################################
    # Read CSV file containing freeze onset and offset
    # Create Pandas DataFrame
    # Rewrite summary.csv
    #############################################
    
    # Initialize Pandas DataFrame
    df = pd.DataFrame()
    columnName = ['folder_videoname', 'single_animal', 'video_system', 'fz_start_sub1', 'fz_end_sub1', 'fz_start_sub2', 'fz_end_sub2']
    columnType = ['str','bool','str','int_array','int_array','int_array','int_array']
    for i in range(len(columnName)):
        df[columnName[i]] = []
    
    # Search subfolders and append DF    
    print("Step1. Reading CSV files from subfolders.")
    print("\tProcessing directory: ", end = " ")
    for dir_name in os.listdir(path):
        if dir_name[0:1] != "_":
            path1 = os.path.join(path, dir_name)
            if os.path.isdir(path1):
                print("{}, ".format(dir_name), end = " ")
                for file in os.listdir(path1):   
                    base = os.path.splitext(file)[0]
                    extn = os.path.splitext(file)[1]
                    if extn == '.csv' and base[0] !='_':
                        filename = os.path.join(path1,file)

                        # Read CSV file
                        # print(filename)
                        single_animal, video_system, sub1Start, sub1End, sub2Start, sub2End = _read_csv(filename)

                        # Store in PD dataframe
                        dir_name_base = dir_name + '_' + base
                        df = df.append({columnName[0]:dir_name_base, columnName[1]:single_animal, columnName[2]:video_system,
                                                  columnName[3]:sub1Start,columnName[4]:sub1End,
                                                  columnName[5]:sub2Start,columnName[6]:sub2End},
                                                   ignore_index=True)
    print("completed.")
    
    # Output to summary.csv
    print("\tWriting summary.csv.")
    write_pd2csv(path,'summary.csv', df, columnName, columnType, 1000)

    #############################################
    # Read summary.csv
    # Compute % freezing and store in DF
    # Output to summary1.csv
    #############################################

    # Read CSV into pandas DF
    df = read_csv2pd(path,'summary.csv', columnName, columnType)
    
    # Compute % freezing and store in DF
    print("\nStep2. Computing %_freezing.")

    columnName = np.append(columnName, ['fz_sub1', 'fz_sub2', 'fz_overlap'])
    columnType = np.append(columnType, ['float','float','float'])

    sub1Freeze = np.zeros(len(df))
    sub2Freeze = np.zeros(len(df))
    overlapFreeze = np.zeros(len(df))

    for i in range (0, len(df)):
        subfolder = os.path.join(path, df.iloc[i,0])
        (sub1Freeze[i], sub2Freeze[i], overlapFreeze[i], overlap) = overlap_freezing(df.iloc[i,:], df['video_system'][i], subfolder, False)
        if df['single_animal'][i] == 'TRUE':
            sub2Freeze[i] = "nan"
            overlapFreeze[i] = "nan"
            
    # Add columns & data
    _df = pd.DataFrame()
    i = 7
    _df[columnName[i]] = sub1Freeze
    _df[columnName[i+1]] = sub2Freeze
    _df[columnName[i+2]] = overlapFreeze
    df = df.join(_df)
    
    # Output to summary1.csv
    print("\tWriting summary1.csv.")
    write_pd2csv(path,'summary1.csv', df, columnName, columnType, 1000)

    #############################################
    # Read summary1.csv
    # Compute permutation/Cohen_D and store in DF
    # Output to summary2.csv
    #############################################

    # Read CSV into pandas DF
    df = read_csv2pd(path, 'summary1.csv', columnName, columnType)
    
    # Compute permutation/Cohen_D and store in DF
    print("\nStep3. Computing permutation/Cohen_D and store in DF.")
    print("\tProcessing column: ", end = " ")
    columnName = np.append(columnName, ['cohen_d'])
    columnType = np.append(columnType, ['float'])

    Cohen_D = np.zeros(len(df))
    
    for i in range (0, len(df)):
        print("{}/{}, ".format(i,len(df)), end=" ")
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            #print("hit!")
            Cohen_D[i] = "nan"
        else:
            subfolder = os.path.join(path, df.iloc[i,0])
            Cohen_D[i] = permutation(df.iloc[i,:], df['video_system'][i], subfolder, False)                                  
    print("completed.")
            
    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[10]] = Cohen_D
    df = df.join(_df)
    
    # Output to summary2.csv
    print("\tWriting summary2.csv.")
    write_pd2csv(path,'summary2.csv', df, columnName, columnType, 1000)    

    #############################################
    # Read summary2.csv
    # Compute lag times
    # Output to summary3.csv
    #############################################

    # Read CSV into pandas DF
    df = read_csv2pd(path, 'summary2.csv', columnName, columnType)
    
    # Compute permutation/Cohen_D and store in DF
    print("\nStep4. Computing lag times.")

    columnName = np.append(columnName, ['lagt_start_s1_s2','lagt_start_s2_s1','lagt_end_s1_s2','lagt_end_s2_s1'])
    columnType = np.append(columnType, ['int_array','int_array','int_array','int_array'])

    s1_s2_start = np.empty((len(df),),dtype=object)
    s2_s1_start = np.empty((len(df),),dtype=object)
    s1_s2_end = np.empty((len(df),),dtype=object)
    s2_s1_end = np.empty((len(df),),dtype=object)
    
    for i in range (0, len(df)):
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            #print("hit!")
            s1_s2_start[i],s2_s1_start[i],s1_s2_end[i],s2_s1_end[i] = np.array([]),np.array([]),np.array([]),np.array([])
        else:
            s1_s2_start[i],s2_s1_start[i],s1_s2_end[i],s2_s1_end[i] = lag_time(df.iloc[i,:], DEBUG=False)

    # Add columns & data
    _df = pd.DataFrame()
    i=11
    _df[columnName[i]]  = s1_s2_start
    _df[columnName[i+1]] = s2_s1_start
    _df[columnName[i+2]] = s1_s2_end
    _df[columnName[i+3]] = s2_s1_end 
    df = df.join(_df)
    
    # Output to summary3.csv
    print("\tWriting summary3.csv.")
    write_pd2csv(path,'summary3.csv', df, columnName, columnType, 1000)   
    
    #############################################
    # Read summary3.csv
    # Count behavioral state & transitions for Markov chain analysis
    # Output to summary4.csv
    #############################################

    # Read CSV into pandas DF
    df = read_csv2pd(path, 'summary3.csv', columnName, columnType)
    
    # Count behavioral state transitions for Markov chain analysis
    print("\nStep5. Counting behavioral state & stansitions for Markov chain analysis")
    print("\tProcessing column: ", end = " ")
    
    columnName = np.append(columnName, ['s_count_0','s_count_1','s_count_2','s_count_3',
                                        'st_count_00','st_count_01','st_count_02','st_count_03',
                                       'st_count_10','st_count_11','st_count_12','st_count_13',
                                       'st_count_20','st_count_21','st_count_22','st_count_23',
                                       'st_count_30','st_count_31','st_count_32','st_count_33'])

    columnType = np.append(columnType, ['int','int','int','int',
                                        'int','int','int','int',
                                        'int','int','int','int',
                                        'int','int','int','int',
                                        'int','int','int','int'])
    
    state_count = np.zeros((len(df),4),dtype=int)
    state_trans_count = np.zeros((len(df),16),dtype=int)
    
    for i in range (0, len(df)):
        print("{}/{}, ".format(i,len(df)), end=" ")
        
        if df['single_animal'][i] == 'TRUE' or df['fz_start_sub1'][i].size == 0 or df['fz_start_sub2'][i].size == 0:
            #print("hit!")
            #state_trans_count[i,:] = np.array([])
            pass
        else:
            state_count[i,:], state_trans_count[i,:] = state_trans(df.iloc[i,:], df['video_system'][i], DEBUG=False)
            
    print("completed.")
    
    # Add columns & data
    _df = pd.DataFrame()    
    for i in range(4):
        _df[columnName[i+15]]  = state_count[:,i]
    for i in range(16):
        _df[columnName[i+19]]  = state_trans_count[:,i]
    df = df.join(_df)    
    
    # Output to summary3.csv
    print("\tWriting summary4.csv.")
    write_pd2csv(path,'summary4.csv', df, columnName, columnType, 1000)   
        
    #############################################
    # Debugging commands
    if DEBUG:
        print(df.dtypes)
        print(df)
        for i in range (0, len(df)):
            for j in range (0, len(df.columns)):
                print(type(df.iloc[i,j]))

    if not DEBUG:
        filename = os.path.join(path,'summary.csv')
        os.remove(filename)
        filename = os.path.join(path,'summary1.csv')
        os.remove(filename)
        filename = os.path.join(path,'summary2.csv')
        os.remove(filename)
        filename = os.path.join(path,'summary3.csv')
        os.remove(filename)   
    return(df)

###############################################################################
def state_trans(df, video_system, DEBUG=False):
    import numpy as np
    import pandas as pd
    
    
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

    # Create working numpy matrix of freezing (True/False) for each video frame
    row, column = 721, 2 # for frame 0-720
    sub_freeze = np.zeros((row,column),dtype=bool) # All values are False
    
    # Mark True when freeze
    # For animal#1
    for i in range(0,int(len(sub1Start))):
        if sub1Start[i] < preStartFrame:
            sub1Start[i] = preStartFrame
        if sub1End[i] > csEndFrame:
            sub1End[i] = csEndFrame
        for j in range(sub1Start[i],sub1End[i]+1):
            sub_freeze[j][0] = True

    # For animal#2
    for i in range(0,int(len(sub2Start))):
        if sub2Start[i] < preStartFrame:
            sub2Start[i] = preStartFrame
        if sub2End[i] > csEndFrame:
            sub2End[i] = csEndFrame
        for j in range(sub2Start[i],sub2End[i]+1):
            sub_freeze[j][1] = True    
    
    # Initialize Pandas DataFrame as df. The size is the same with sub_freeze (frame 0-720)
    _df = pd.DataFrame(data=sub_freeze, columns=['sub1_freeze', 'sub2_freeze'], dtype=bool)
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

    for i in range (0, len(_df)):
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

    for i in range (0, len(_df)-1):
        new_row = {'From':_df['state'][i], 'To':_df['state'][i+1]}
        _df2 = _df2.append(new_row, ignore_index=True)

    #################################################
    # Search for each of 16 patterns
    #   during CS (either 241-720 or 240-719). Frame number
    #   is 480, but the number of states is 479)
    #################################################
    pattern = [[0, 0],[0, 1],[0, 2],[0, 3],
               [1, 0],[1, 1],[1, 2],[1, 3],
               [2, 0],[2, 1],[2,2 ],[2, 3],
               [3, 0],[3, 1],[3, 2],[3, 3]]
    state_trans_count = np.zeros(16, dtype=int)

    for i in range (csStartFrame, csEndFrame):
        for j in range (len(pattern)):
            if _df2['From'][i] == pattern[j][0] and _df2['To'][i] == pattern[j][1]:
                state_trans_count[j] += 1
                
    # print(state_trans_count)
                
    return(state_count,state_trans_count)

###############################################################################
def lagtime(w1, w2, DEBUG=False):
    # Search the closest epochs from partner (w2)
    # lag time > 0 means w2 follows w1
    
    import numpy as np
    
    indexCloseset = np.zeros(len(w1),dtype=int) # Indices of the closest freezing epoch from partner
    lagTime = np.zeros(len(w1),dtype=int)   # lag time (frame number)

    for i in range(0,len(w1)):
        _lagTime = 10000
        _indexCloseset = 0
        # Serach in w2
        for j in range(0,len(w2)):      
            if abs(_lagTime) > abs(w2[j] - w1[i]) :
                _indexCloseset = j
                _lagTime = w2[j] - w1[i]

        indexCloseset[i] = _indexCloseset
        lagTime[i] = _lagTime
    
    if DEBUG:
        print("Sub1 epoch ID     :", end =" ")
        print(*range(0,len(w1)), sep = ", ")

        print("Sub1 frame number :", end =" ") 
        print(*w1, sep = ", ")

        print("Sub2 freeze epoch :", end =" ")          
        print(*indexCloseset, sep = ", ")

        print("Sub2 frame number :", end =" ")
        print(*w2[indexCloseset], sep = ", ")    

        print("lag-time          :", end =" ")        
        print(*lagTime, sep = ", ")
        
    return(lagTime)


def lag_time(df, DEBUG=False):
    
    s1 = df['fz_start_sub1']
    s2 = df['fz_start_sub2']
    s1_s2_start = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

    s1 = df['fz_start_sub2']
    s2 = df['fz_start_sub1']
    s2_s1_start = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

    s1 = df['fz_end_sub1']
    s2 = df['fz_end_sub2']
    s1_s2_end = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

    s1 = df['fz_end_sub2']
    s2 = df['fz_end_sub1']
    s2_s1_end = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

    return(s1_s2_start, s2_s1_start, s1_s2_end, s2_s1_end)

###############################################################################
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
    for i in range(0,len(rows)):
        if rows[i][0] == 'single_animal:':
            single_animal = rows[i][1] 

    # Detect video system, either FreezeFrame or PCBox
    video_system = ''
    for i in range(0,len(rows)):
        if rows[i][0] == 'video:':
            video_system = rows[i][1]            

    # Read start and end for subject1
    # for i in range(2,len(rows)):
    _data_start = False
    for i in range(0,len(rows)):
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
    for i in range(0,len(rows)):
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
        
    return (single_animal,video_system,sub1Start,sub1End,sub2Start,sub2End)

###############################################################################
def read_csv(filename):

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

    # Ignore the first two rows. Read until blank comes up
    # Read start and end for subject1
    for i in range(2,len(rows)):
        if rows[i][0] == '':
            break
        _sub1Start.append(int(rows[i][0]))
        _sub1End.append(int(rows[i][1]))

    # Read start and end for subject2
    for i in range(2,len(rows)):
        if rows[i][3] == '':
            break
        _sub2Start.append(int(rows[i][3]))
        _sub2End.append(int(rows[i][4]))

    # convert to numpy array
    import numpy as np
    sub1Start = np.array(_sub1Start)
    sub1End = np.array(_sub1End)
    sub2Start = np.array(_sub2Start)
    sub2End = np.array(_sub2End)
        
    return (sub1Start,sub1End,sub2Start,sub2End)

##################################################################################################
def overlap_freezing (df, video_system, path, output):
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

    
    filename = os.path.join(path,'overlap_fig.eps')
    
    video_system = df['video_system']

    sub1Start = df['fz_start_sub1']
    sub1End = df['fz_end_sub1']
    sub2Start = df['fz_start_sub2']
    sub2End = df['fz_end_sub2']
    
#     i = 3
#     sub1Start = df[i]
#     sub1End = df[i+1]
#     sub2Start = df[i+2]
#     sub2End = df[i+3]
    
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

    # Create working numpy matrix of freezing (True/False) for each video frame
    column, row = 3, 721;
    overlap = np.array([[0 for x in range(column)] for y in range(row)])

    # Set the overlap as 1 at freeze for each animal
    # For animal#1
    for i in range(0,int(len(sub1Start))):
        if sub1Start[i] < preStartFrame:
            sub1Start[i] = preStartFrame
        if sub1End[i] > csEndFrame:
            sub1End[i] = csEndFrame
        for j in range(sub1Start[i],sub1End[i]+1):
            overlap[j][0] = 1

    # For animal#2
    for i in range(0,int(len(sub2Start))):
        if sub2Start[i] < preStartFrame:
            sub2Start[i] = preStartFrame
        if sub2End[i] > csEndFrame:
            sub2End[i] = csEndFrame        
        for j in range(sub2Start[i],sub2End[i]+1):
            overlap[j][1] = 1

    # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
    # and overlapped freezing (counter[2])
    counter = np.zeros((3), dtype=int)

    #for i in range(241,int(len(overlap))):
    for i in range(csStartFrame,csEndFrame + 1):
            if overlap[i,0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i,1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i,0] == 1 and overlap[i,1] == 1:
                overlap[i,2] = 1
                counter[2] = counter[2] + 1

    sub1Freeze = counter[0]/480.0*100.0
    sub2Freeze = counter[1]/480.0*100.0
    overlapFreeze = counter[2]/480.0*100.0
    
    # output
    if output:
        print("Folder name: " + df[0]) 
        print("Animal1 freeze : %f" % (counter[0]/480.0*100.0))
        print("Animal2 freeze : %f" % (counter[1]/480.0*100.0))
        print("Overlap freeze : %f" % (counter[2]/480.0*100.0))

        # Plotting the freezing dynamics
        fig = plt.figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
        fig.subplots_adjust(top=0.8)

        ax1 = fig.add_subplot(211)
        x = overlap[:,0] + 1.75
        y = overlap[:,1] + 1.25
        z = overlap[:,2]

        ax1.plot(x) 
        ax1.plot(y) 
        ax1.plot(z) 

        ax1.set_xlabel('x - axis')  
        ax1.set_ylabel('y - axis') 
        ax1.set_title('From top: Animal1, Animal2 and overlap!')   
        plt.savefig(filename, format='eps', dpi=1000)
        
    return(sub1Freeze, sub2Freeze, overlapFreeze, overlap)


##################################################################################################
def permutation(df, video_system, path, output):
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
    (_sub1Freeze, _sub2Freeze, _overlapFreeze, overlap) = overlap_freezing(df, video_system, path, False)

    # Permutation
    # Repeat random shift of subject1 for 1000 times 
    nRepeat = 1000

    sub1Freeze = np.array([0.0 for x in range(nRepeat)])
    sub2Freeze = np.array([0.0 for x in range(nRepeat)])
    overlapFreeze = np.array([0.0 for x in range(nRepeat)])

    for x in range(nRepeat):
        # Generate random number ranged from 0 to 479
        shift = random.randint(0,479)
        # Shift the freezing pattern in subject1 only during frame 241-720
        overlap[241:720,0] = np.roll(overlap[241:720,0],shift)    

        # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
        # and overlapped freezing (counter[2])
        counter = np.zeros((3), dtype=int)

        for i in range(241,len(overlap)):
            if overlap[i,0] == 1:
                counter[0] = counter[0] + 1
            if overlap[i,1] == 1:
                counter[1] = counter[1] + 1
            if overlap[i,0] == 1 and overlap[i,1] == 1:
                overlap[i,2] = 1
                counter[2] = counter[2] + 1
            else:
                overlap[i,2] = 0

        sub1Freeze[x] = counter[0]/480.0*100.0
        sub2Freeze[x] = counter[1]/480.0*100.0
        overlapFreeze[x] = counter[2]/480.0*100.0

    Cohen_D = (_overlapFreeze - np.mean(overlapFreeze)) / np.std(overlapFreeze)

    # Output to permutation.csv
    if output:    
        print("\tWriting permutation.csv.")
        outputFilename = os.path.join(path,"_permutation.csv")
        print("\t\t" + outputFilename)
        output = open(outputFilename,"w")

        output.write('sub1Freeze, sub2Freeze, overlapFreeze\n')
        output.write(
                str(_sub1Freeze) + ',' +
                str(_sub2Freeze) + ',' +
                str(_overlapFreeze) + '\n')

        for i in range (0, len(sub1Freeze)):
            output.write(
                str(sub1Freeze[i]) + ',' +
                str(sub2Freeze[i]) + ',' +
                str(overlapFreeze[i]) + ',' +
                'Permutation' + '\n')
        output.close()

        print("\tObserved overlap: {} \n\tTheoretical random overlap (mean): {} (SD): {} \n\tCohen_D: {}".format(
            _overlapFreeze, np.mean(overlapFreeze), np.std(overlapFreeze), Cohen_D))
               
    return(Cohen_D)
##################################################################################################
def write_pd2csv(path,filename,df,columnName,columnType,mlw=1000):
    import os
    import numpy as np
    import pandas as pd
    
    outputFilename = os.path.join(path,filename)
    output = open(outputFilename,"w")
    # mlw = 1000 # max_line_width in np.array2string
    
    output.write(','.join(columnName)+'\n')

    for i in range (0, len(df)):
        output_str = ''
        for j in range (0, len(columnName)):
            # print(df.shape,j)
            output_str = preprocess_output_str(output_str, df.iloc[i,j], columnType[j], 1000)
        output.write(output_str[0:-1] + '\n')
    output.close()
    return
    
def preprocess_output_str(output_str, data, columnType, mlw=1000):
    import numpy as np

    if columnType == 'int_array':
        output_str = output_str + np.array2string(data,max_line_width=mlw) + ','
    elif columnType == 'float' or columnType == 'bool' or columnType =='int':
        output_str = output_str + str(data) + ','
    elif columnType =='str':
        output_str = output_str + data + ','

    return(output_str)

##################################################################################################
def read_csv2pd(path,filename,columnName,columnType):
    import os
    import numpy as np
    import pandas as pd

    inputFilename = os.path.join(path,filename)

    df = pd.read_csv(inputFilename,index_col=False)
    df = df.astype(object) # Need to convert object to set numpy array in a cell
        
    # Post process from str to array
    for i in range (0, len(df)):
        for j in range (0, len(df.columns)):
            if columnType[j] == 'int_array':
                df.iloc[i,j] = np.fromstring(df.iloc[i,j][1:-1],dtype=int,sep=' ')

    return(df)

##################################################################################################