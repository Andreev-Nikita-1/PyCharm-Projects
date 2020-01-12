import os

dir = '/home/nikita/tracking/pool_9965746_12-01-2020_12-00-20'
dir0 = '/home/nikita/tracking/2'
files1 = os.listdir(dir)
f = open("commands2.txt", 'w')
formats = ['mp3', 'amr', 'wma', 'wav', 'm4a', '3gpp', 'aac', 'mp4', 'ogg']
for folder in files1:
    try:
        os.mkdir(dir0 + '/' + folder)
    except:
        pass
    files = os.listdir(dir + '/' + folder)
    s1 = 'ffmpeg -i '
    s2 = ' -acodec pcm_s16le -ac 1 -ar 16000 '
    if (files[0][-3:]) in formats:
        n = 4
    elif files[0][-4:] == '3gpp':
        n = 5
    elif files[0][-4:] == '.exe' or files[0][-4:] == '.jpg':
        print(' exe', folder)
        n = -1
    else:
        n = 0
    for file in files:
        if n == -1:
            break
        if n == 0:
            filename = file + '.wav'
        else:
            filename = file[:-n] + '.wav'
        f.write(s1 + dir + '/' + folder + '/' + file + s2 + dir0 + '/' + folder + '/' + filename + '\n')
f.close()
