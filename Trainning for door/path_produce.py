#
#this file is used to produce the data patchs. 
#
#


import os
import cPickle
import pickle
import numpy as np
from numpy import array, append
from PIL import Image


def makeBatch (load_path, save_path, data_size):
    data = []
    filenames = []
    class_list = []
    class_file = file('train-origin-pics-labels.txt', 'r+').readlines()
    file_list = os.listdir(load_path)
    num_sq = save_path[len(save_path)-1]
    for item in  file_list:
        if item.endswith(".jpg"):
            picture_number = item[0:len(item)-4]
            picture_num = int(picture_number)
            class_picture = class_file[picture_num-1][10:11]
            if int(picture_num)%100 == 0:
                print picture_number
            n = os.path.join(load_path, item)
            inputImage = Image.open(n)
            (width,height) = inputImage.size
            #if  width > height:
            #    newwidth = width/height*128
            #    small_image = inputImage.resize((newwidth, 128),Image.ANTIALIAS)
            #else:
            #    newheight = height/width*128
            #    small_image = inputImage.resize((128, newheight),Image.ANTIALIAS)
            small_image = inputImage.resize((data_size, data_size),Image.ANTIALIAS)
            try:
                r, g, b = small_image.split()
                reseqImage = list(r.getdata()) + list(g.getdata()) + list(b.getdata())
                data.append(reseqImage)
                filenames.append(item)
                class_list.append(class_picture)
            except:
                print 'error' + picture_number
    data_array = np.array(data, dtype = np.uint8)
    T_data = data_array.T
    out_file = file(save_path, 'w+')
    dic = {'batch_label':'batch ' + num_sq + ' of 6', 'data':T_data, 'labels':class_list, 'filenames':filenames}
    pickle.dump(dic, out_file)
    out_file.close()

def read_batch(batch_path, data_size):
    in_file = open(batch_path, 'r+')
    xx = cPickle.load(in_file)
    in_file.close()
    T_datas = xx['data']
    datas = T_datas.T
    c = np.zeros((1, data_size*data_size*3), dtype=np.float32)
    i  = 0
    for data in datas:
        i += 1
        c = c + data
    return i, c

def add_all(data_size, path):
    count = 0
    totalc = np.zeros((1, data_size*data_size*3), dtype=np.float32)
    for idx in range(1, 7):
        print 'reading batch'+str(idx)
        path += '/data_batch_' + str(idx)
        curcount, curc = read_batch(path, data_size)
        count += curcount
        totalc = totalc + curc

    return count, totalc

def write_data(data_size, path):
    cout, total = add_all(data_size)
    a  = []
    for i in range(0, len(total[0])):
        c = total[0][i] / cout
        a.append( [c])
    a_array = array(a, dtype = np.float32)
    return a_array

def main(data_size, path):
    data_mean = write_data(data_size, path)
    label_names = ['nopeople', 'exist_people']
    num1 = 5000
    num2 = data_size*data_size*3
    dic = {'data_mean':data_mean, 'label_names':label_names, 'num_cases_per_batch':num1, 'num_vis':num2}
    out_file = open(path+'/batches.meta', 'w+')
    cPickle.dump(dic, out_file)
    out_file.close()

data_size = 64
for i in range(1, 7):
    makeBatch('./train-origin-pics-part'+str(i), 'baidu_data_size_'+str(data_size)+'/data_batch_'+str(i), data_size)
main(data_size, 'baidu_data_size_'+str(data_size))
