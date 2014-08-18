__author__ = 'zhyf'
import numpy as np
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
from PIL import Image 
# import Image

class ShowNetError(Exception):
    pass


class ShowPredction(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)

    def get_gpus(self):
        self.need_gpu = self.op.get_value('show_preds') or self.op.get_value('write_features')
        if self.need_gpu:
            ConvNet.get_gpus(self)
        print 'finish_0'

    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
        # self.train_data_provider = self.test_data_provider = Dummy()
        print 'finish_1'

    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
        print 'finish_2'

    def init_model_state(self):
        #ConvNet.init_model_state(self)
        if self.op.get_value('show_preds'):
            self.sotmax_idx = self.get_layer_idx(self.op.get_value('show_preds'), check_type='softmax')
        if self.op.get_value('write_features'):
            self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('write_features'))
        print 'finish_3'

    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
        print 'finish_4'

    def show_predictions(self, pic_path):
        preds = n.zeros((1, 6), dtype=n.single)
        data = []
        file1 = file('.//storage2//door//door_data60//batches.meta', 'rb')
        meta_cifa = cPickle.load(file1)
        file1.close()
        data_size = int(sqrt((meta_cifa['data_mean'].size)/3))
        # print data_size
        inputImage = Image.open(pic_path)

        small_image = inputImage.resize((data_size, data_size),Image.ANTIALIAS)
        try:
            r, g, b = small_image.split()
            reseqImage = list(r.getdata()) + list(g.getdata()) + list(b.getdata())
            # print reseqImage
        except:
            return '1'
        pixel_image = []
        for pixel in reseqImage:
            pixel_image.append([pixel])
        data_array = np.array(pixel_image, dtype = np.float32)
        data.append(n.require((data_array - meta_cifa['data_mean']), dtype=n.single, requirements='C'))
        filepath,filename = os.path.split(pic_path)

        print filename,model.ClassNameTest(filename)
        data.append(n.require((np.array([[model.ClassNameTest(filename)]])), dtype=n.single, requirements='C'))
        print n.require((np.array([[model.ClassNameTest(filename)]])))
        data.append(preds)
        print data[0].shape,data[1].shape,data[2].shape
        # print data
        # temp = data[0]
        # print temp.ndim
        # print temp.shape,temp.size
        self.libmodel.startFeatureWriter(data, self.sotmax_idx)
        self.finish_batch()
        label_names = meta_cifa['label_names']
        img_labels = sorted(zip(preds[0,:], label_names), key=lambda x: x[0])[-1:]
        # print img_labels

        if img_labels[0][1] == 'Right 14':
            x = '5'
        elif img_labels[0][1] == 'Right 18.4':
            x = '4'
        elif img_labels[0][1] == 'Front':
            x = '3'
        elif img_labels[0][1] == 'Left 18.4':
            x = '2'
        elif img_labels[0][1] == 'Left 14':
            x = '1'
        elif img_labels[0][1] == 'No door':
            x = '0'
        return x
        print 'finish_5'

    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range'):
                op.delete_option(option)
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")

        op.options['load_file'].default = None
        op.options['load_file'].value_given = False
        return op
        print 'finish_6'
    def ClassNameTest(self,item):
        if item.endswith("testr2p.jpg"):
            return '5'
        elif item.endswith("testr1p.jpg"):
            return '4'
        elif item.endswith("test0p.jpg"):
            return '3'
        elif item.endswith("testl2p.jpg"):
            return '2'
        elif item.endswith("testl1p.jpg"):
            return '1'
        else:
            return '0'
        print 'finish_7'
    # def ClassNameTest(item):
    # if item.endswith("testp.jpg"):
    #     return [1]
    # else:
    #     return [0]

def show_predict_dir(load_path):
    try:
        error = 0
        file_list = os.listdir(load_path)
        result = []
        # class_file = file('./baidu_result/train-origin-pics-labels.txt', 'rb').readlines()
        i=0
        P_num=0
        for item in file_list:
            i=i+1;
            if item.endswith('.jpg'):
                # picture_number = item[0:len(item)-4]
                # picture_index = int(picture_number) - 1
                # if picture_index % 1000 == 0:
                #     print picture_index
                n = os.path.join(load_path, item)
                #print item
                door = model.show_predictions(n)
                # for l_lable in door_label:
                result.append(item + ';' +model.ClassNameTest(item)+';'+ door + '\n')
                ground_truth=model.ClassNameTest(item)
                print ground_truth,door
                if not model.ClassNameTest(item)=='0':
                    P_num +=1

                if not door == ground_truth:
                    error += 1
        erro_ratio = float(error)/len(result)
        print erro_ratio
        print i,P_num,len(result),error
        result.append('error_ratio:'+str(erro_ratio)+' Positive_num:'+str(P_num)+' total_num:'+str(i))
        myreslut = sorted(result, key=lambda result:result[0])
        if P_num<2000:
            my_result = file('myresult_p.txt', 'wb')
        else:
            my_result = file('myresult_n.txt', 'wb')
        my_result.writelines(myreslut)
        my_result.close()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e
    print 'finish_8'


op = ShowPredction.get_options_parser()
op, load_dic = IGPUModel.parse_options(op)
model = ShowPredction(op, load_dic)
print os.path.exists("G:\\door_data_sampling\\posture\\data_pos\\test\\test_value_p\\")
show_predict_dir('G:\\door_data_sampling\\posture\\data_pos\\test\\test_value_p\\')
