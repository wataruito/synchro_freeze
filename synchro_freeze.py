# extracting onset and offset of freezing from two subject from a csv file
'''
File format for the csv file:
Currently frame number and sec appear together. Need to clean up.

animal1						animal2		
start	end		duration	start	end		duration
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
					
	total (s)	71.5						79
		%		59.58333333					65.83333333
'''

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

#     s1 = df['Sub1_start'][0]
#     s2 = df['Sub2_start'][0]
#     s1_s2_start = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

#     s2 = df['Sub1_start'][0]
#     s1 = df['Sub2_start'][0]
#     s2_s1_start = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

#     s1 = df['Sub1_end'][0]
#     s2 = df['Sub2_end'][0]
#     s1_s2_end = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2

#     s2 = df['Sub1_end'][0]
#     s1 = df['Sub2_end'][0]
#     s2_s1_end = lagtime(s1,s2,DEBUG) # Freezing onset from s1 mouse to s2
    
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
def overlap_freezing (df, path, output):
    # Compute overlap of freezing
    #
    # The original videos is at 4 frame per sec (0.25s/frame)
    # The duration is 3 min: total 720 frames
    #     1 min for acclimation (frame 1-240)
    #     2 min for CS (frame 241-720))
    # 
    #     Create np.array of 721 rows x 2 columes, 
    #     representing two mice and 720 video frames (1 to 720 frame)
    #     column 0 will be not used.
    
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    sub1Start = df[1]
    sub1End = df[2]
    sub2Start = df[3]
    sub2End = df[4]
    
    filename = os.path.join(path,'overlap_fig.eps')
        
    column, row = 3, 721;
    overlap = np.array([[0 for x in range(column)] for y in range(row)])

    # Set the overlap as 1 at freeze for each animal
    # For animal#1
    for i in range(0,int(len(sub1Start))):
        for j in range(sub1Start[i],sub1End[i]+1):
            overlap[j][0] = 1

    # For animal#2
    for i in range(0,int(len(sub2Start))):
        for j in range(sub2Start[i],sub2End[i]+1):
            overlap[j][1] = 1

    # Scan the overlap valiable for freezing in animal#1 (counter[0]), animal#2 (counter[1])
    # and overlapped freezing (counter[2])
    counter = np.zeros((3), dtype=int)

    for i in range(241,int(len(overlap))):        
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
def permutation(df, path, output):
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
    (_sub1Freeze, _sub2Freeze, _overlapFreeze, overlap) = overlap_freezing(df, path, False)

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
    columnName = ['folder_videoname', 'fz_start_sub1', 'fz_end_sub1', 'fz_start_sub2', 'fz_end_sub2']
    columnType = ['str','int_array','int_array','int_array','int_array']
    for i in range(len(columnName)):
        df[columnName[i]] = []
    
    # Search subfolders and append DF    
    print("Step1. Reading CSV files from subfolders.")
    for dir_name in os.listdir(path):
        path1 = os.path.join(path, dir_name)
        if os.path.isdir(path1): 
            for file in os.listdir(path1):   
                base = os.path.splitext(file)[0]
                extn = os.path.splitext(file)[1]
                if extn == '.csv' and base[0] !='_':
                    filename = os.path.join(path1,file)
                    
                    # Read CSV file
                    print("\tProcessing directory: {}".format(dir_name))
                    sub1Start, sub1End, sub2Start, sub2End = read_csv(filename)
                                        
                    # Store in PD dataframe
                    dir_name_base = dir_name + '_' + base
                    df = df.append({columnName[0]:dir_name_base,
                                              columnName[1]:sub1Start,columnName[2]:sub1End,
                                              columnName[3]:sub2Start,columnName[4]:sub2End},
                                               ignore_index=True)
    
    # Output to summary.csv
    print("\tWriting summary.csv.")
    write_pd2csv(path,'summary.csv', df, columnName, columnType, 1000)

    #############################################
    # Read summary.csv
    # Compute % freezing and store in DF
    # Output to summary1.csv
    #############################################

    # Read CSV into pandas DF
    read_csv2pd(path,'summary.csv', df, columnName, columnType)
    
    # Compute % freezing and store in DF
    print("\nStep2. Computing %_freezing.")

    columnName = np.append(columnName, ['fz_sub1', 'fz_sub2', 'fz_overlap'])
    columnType = np.append(columnType, ['float','float','float'])

    sub1Freeze = np.zeros(len(df))
    sub2Freeze = np.zeros(len(df))
    overlapFreeze = np.zeros(len(df))

    for i in range (0, len(df)):
        subfolder = os.path.join(path, df.iloc[i,0])
        (sub1Freeze[i], sub2Freeze[i], overlapFreeze[i], overlap) = overlap_freezing(df.iloc[i,:], subfolder, False)

    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[5]] = sub1Freeze
    _df[columnName[6]] = sub2Freeze
    _df[columnName[7]] = overlapFreeze
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
    read_csv2pd(path, 'summary1.csv', df, columnName, columnType)
    
    # Compute permutation/Cohen_D and store in DF
    print("\nStep3. Computing permutation/Cohen_D and store in DF.")

    columnName = np.append(columnName, ['cohen_d'])
    columnType = np.append(columnType, ['float'])

    Cohen_D = np.zeros(len(df))
    
    for i in range (0, len(df)):
        subfolder = os.path.join(path, df.iloc[i,0])
        Cohen_D[i] = permutation(df.iloc[i,:], subfolder, False)

    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[8]] = Cohen_D
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
    read_csv2pd(path, 'summary2.csv', df, columnName, columnType)
    
    # Compute permutation/Cohen_D and store in DF
    print("\nStep4. Computing lag times.")

    columnName = np.append(columnName, ['lagt_start_s1_s2','lagt_start_s2_s1','lagt_end_s1_s2','lagt_end_s2_s1'])
    columnType = np.append(columnType, ['int_array','int_array','int_array','int_array'])

    s1_s2_start = np.empty((len(df),),dtype=object)
    s2_s1_start = np.empty((len(df),),dtype=object)
    s1_s2_end = np.empty((len(df),),dtype=object)
    s2_s1_end = np.empty((len(df),),dtype=object)
    
    for i in range (0, len(df)):
        s1_s2_start[i],s2_s1_start[i],s1_s2_end[i],s2_s1_end[i] = lag_time(df.iloc[i,:], DEBUG=False)

    # Add columns & data
    _df = pd.DataFrame()
    _df[columnName[9]]  = s1_s2_start
    _df[columnName[10]] = s2_s1_start
    _df[columnName[11]] = s1_s2_end
    _df[columnName[12]] = s2_s1_end 
    df = df.join(_df)
    
    # Output to summary2.csv
    print("\tWriting summary3.csv.")
    write_pd2csv(path,'summary3.csv', df, columnName, columnType, 1000)   
    
    
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
    return(df)

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
    elif columnType == 'float':
        output_str = output_str + str(data) + ','
    elif columnType =='str':
        output_str = output_str + data + ','

    return(output_str)

##################################################################################################
def read_csv2pd(path,filename,df,columnName,columnType):
    import os
    import numpy as np
    import pandas as pd

    inputFilename = os.path.join(path,filename)

    df = pd.read_csv(inputFilename,index_col=False)

    # Post process from str to array
    for i in range (0, len(df)):
        for j in range (0, len(df.columns)):
            if columnName[j] == 'int_array':
                df.iloc[i,j] = np.fromstring(df.iloc[i,j][1:-1],dtype=int,sep=' ')

    return(df)

##################################################################################################