import numpy as np
import cv2
import mxnet as mx
from glob import glob
from skimage import transform as trans
from utils.detector import Detector
from utils.search_engine import Search_Engine

class Extractor(object):
    def __init__(self, prefix, epoch, gpu_list=(0, ), batch_size=32, out_name='fc1_output'):
        ctx = []
        if len(gpu_list) == 0:
            ctx = [mx.cpu()]
        else:
            for gpu_id in gpu_list:
                ctx.append(mx.gpu(gpu_id))

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[out_name]
        self.shape = (batch_size, 3, 112, 112)
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (batch_size, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)

    def forward(self, data):
        self.model.forward(data, is_train=False)
        fea = self.model.get_outputs()[0].asnumpy()
        return fea

class Gender:
    def __init__(self):
        ga_model = ['./model/ga-model/model', '00']
        image_size = [112, 112, 3]
        prefix = ga_model[0]
        epoch = int(ga_model[1])
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.model = mx.mod.Module(symbol=sym, context=mx.gpu(int(0)), label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    def gen_age(self,image_list):
        ga_list = []
        for index, value in enumerate(image_list):
            img = image_list[index]
            input_blob = np.expand_dims(img, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            ret = self.model.get_outputs()[0].asnumpy()
            g = ret[:, 0:2].flatten()
            gender = np.argmax(g)
            a = ret[:, 2:202].reshape((100, 2))
            a = np.argmax(a, axis=1)
            age = int(sum(a))
            ga_list.append([gender, age])
        return ga_list

class Face_Extractor:
    def __init__(self):
        model_path = './model/mtcnn'
        face_model_path = './model/model-r50-google/model,00'
        prefix, epoch = face_model_path.split(',')
        epoch = int(epoch)
        gpu_list = [0]
        batch_size = 1
        self.detector = Detector(model_path, gpu_fraction=0.5)
        self.extracker = Extractor(prefix, epoch, gpu_list, batch_size)
        self.gender = Gender()


    def preprocess(self,img_list):
        warped_list = []
        bbox_list = []
        for img in img_list:
            img = img[..., ::-1]
            img_size = [112, 112, 3]
            bboxs = self.detector.detect_face(img, debug=False)
            if len(bboxs) >= 1:
                area_list = [bbox['area'] for bbox in bboxs]
                index = area_list.index(max(area_list))
                result = bboxs[index]
                bbox = result['bbox']
                det_lmk = np.array([result['left_eye'], result['right_eye'], result['nose'], result['left_mouth'],
                                    result['right_mouth']])
                src = np.array([
                    [30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)
                if img_size[1] == 112:
                    src[:, 0] += 8.0
                tform = trans.SimilarityTransform()
                tform.estimate(det_lmk, src)
                M = tform.params[0:2, :]
                warped = cv2.warpAffine(img, M, (img_size[1], img_size[0]), borderValue=0.0)
                warped = np.transpose(warped,(2,0,1))
            else:
                warped = None
                bbox = None
            warped_list.append(warped)
            bbox_list.append(bbox)
        return warped_list,bbox_list

    def extract_fea(self,img_list,gender=False):
        fea_list = []
        warped_list, bbox_list = self.preprocess(img_list)
        valid_index = [i for i in range(len(warped_list)) if warped_list[i] is not None]
        valid_image = [warped_list[i] for i in valid_index]
        feas = self.extracker.forward(mx.io.DataBatch(data=(mx.nd.array(valid_image),)))
        assert len(warped_list)==len(bbox_list)==len(img_list),'length not equal,image:{},bbox:{},img:{}'.format(len(warped_list),len(bbox_list),len(img_list))
        if not gender:
            j=0
            for i in range(len(img_list)):
                if warped_list[i] is not None:
                    fea_list.append(feas[j])
                    j+=1
                else:
                    fea_list.append(None)
            assert j==len(valid_image),'feature index not match the image index'
            return bbox_list,fea_list
        if gender:
            ga_list = []
            gender_age = self.gender.gen_age(valid_image)
            j=0
            for i in range(len(img_list)):
                if warped_list[i] is not None:
                    fea_list.append(feas[j])
                    ga_list.append(gender_age[j])
                    j+=1
                else:
                    fea_list.append(None)
                    ga_list.append(None)
            assert j==len(valid_image),'feature index not match the image index'
            return bbox_list,fea_list,ga_list

if __name__ == '__main__':
    face_extractor = Face_Extractor()
    img_path = './images'
    img_list = glob(img_path+'/*.jpg')
    # img_list = ['/home/gehen/PycharmProjects/face/images/Alejandro_Avila_0001.jpg','/home/gehen/PycharmProjects/face/images/Alejandro_Avila_0002.jpg']
    image_list = [cv2.imread(img,cv2.IMREAD_COLOR) for img in img_list]
    # image_list.append(np.zeros([100,100,3]))
    bbox_list,feas_list = face_extractor.extract_fea(image_list)
    # bbox_list2,feas_list2,ga_list2 = face_extractor.extract_fea(image_list,True)
    print feas_list[0]

    ## build the search engine
    assert None not in bbox_list, 'Exist None face images in gallary'
    feas = np.array(feas_list)
    search_engine = Search_Engine(feas)
    search_engine.train()
    image = cv2.imread('/home/gehen/PycharmProjects/face/Anders_Ebbeson_0003.jpg',cv2.IMREAD_COLOR)
    fea1 = face_extractor.extract_fea([image])
    sim, idx = search_engine.search(fea1[1][0])
    print 'person index is {}, and similarity is {}'.format(idx,sim)
    print img_list
