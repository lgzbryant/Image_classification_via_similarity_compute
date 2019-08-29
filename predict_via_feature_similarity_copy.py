# coding=utf-8
import h5py
import numpy as np
import time

# def cosine(q,a):
#     pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
#     pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
#     pooled_mul_12 = tf.reduce_sum(q * a, 1)
#     score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
#     return score


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def read_from_h5py(file):
    '''
    从file中读出存储的特征，并用list返回
    :param file: h5文件file
    :return:get_all_features 一个python list，从h5文件file读出来的特征。
    '''

    get_all_features = []
    # vgg_feature = h5py.File('features.h5', 'r')
    vgg_feature = h5py.File(file, 'r')
    keys = vgg_feature.keys()
    values = vgg_feature.values()

    # print(keys)
    # print(values)
    # print(vgg_feature.items())

    filenames = vgg_feature['filenames']

    for i in range(len(filenames)):
        arr1ev = filenames[i]
        image_feature = vgg_feature["resnet_v1_101/logits"][i]
        # print(arr1ev)
        # print(image_feature.shape)
        image_feature = image_feature.reshape((1, 1000))
        # print(image_feature.shape)
        get_all_features.append(image_feature)

    vgg_feature.close()
    # print(get_all_features[0])
    return get_all_features


def find_threshold(predict_h5, image_h5):
    '''
    用来找可以将predict_h5和image_h5两个文件中装的特征完全区分的一个阈值threshold
    :param predict_h5:
    :param image_h5:
    :return:

    +---for 每次取一个阈值，记为each_threshold，
        ---for  predict_h5中的每个特征，每次记为each_predict,
            ---for  image_h5中的每个特征，每次记为each_image_in_pool,
                 ---如果 each_predict 和 each_image_in_pool 余弦相似度大于该阈值，则each_predict属于image_h5一类；
        ---simi是整体相似率，也就是predict_5的特征有多少被判断为image_h5所代表的类别

    每次取一个阈值，计算对应的simi，随阈值的提高，simi越来越低，为0时代表predict_5中没有任何特征被判断为image_5类别，
    此时，表示在该阈值下，image_h5和predict_h5能够完全分开。

    '''

    predict_total_images = len(predict_h5)
    simi = []
    corresponding_threshold = []


    for each_threshold in np.arange(0.5, 1.0, 0.02):
        in_image_pool = 0
        corresponding_threshold.append(each_threshold)

        order = 1
        for each_predict in predict_h5:
            print('the threshold is {} , and now predict the {}/({}) images'.format(each_threshold, order,
                                                                                    predict_total_images))
            order += 1
            for each_image_in_pool in image_h5:
                if cos_sim(each_predict, each_image_in_pool) >= each_threshold :
                    in_image_pool += 1
                    break

        simi.append(in_image_pool / predict_total_images * 100.0)

    total = len(simi)
    for i in range(total):
        print(
            'when the threshold is {}, these test imgase has {} % protest '.format(corresponding_threshold[i], simi[i]))



def predict_via_feature(predict_h5, image_h5, threshold):
    '''
    计算在取阈值为threshold时，predict_h5中的特征有多少被预测为和image_h5一类。
    :param predict_h5:
    :param image_h5:
    :param threshold:
    :return:

    for 待预测的predict_h5中的每一个特征each_predict;
        for 每一个image_h5库中的each_image_in_pool
            如果each_predict和each_image_in_pool的相似度大于阈值threshold，则属于image_h5类。

    '''

    predict_total_images = len(predict_h5)
    in_image_pool = 0
    order = 1
    for each_predict in predict_h5:
        print('the threshold is {} , and now predict the {}/({}) images'.format(threshold, order,
                                                                                predict_total_images))
        order += 1
        for each_image_in_pool in image_h5:
            if cos_sim(each_predict, each_image_in_pool) >= threshold:
                in_image_pool += 1
                break
    print('when the threshold is {}, these test imgase has {} % protest '.format(threshold, (in_image_pool / predict_total_images * 100.0)))

if __name__ == '__main__':


    begin_time = time.time()
    find_threshold(
        predict_h5=read_from_h5py('features_simi.h5'),
        image_h5=read_from_h5py('features_protest.h5')
    )
    end_time = time.time()
    print('cost time: {} min'.format((end_time - begin_time)/60))


    predict_via_feature(
        predict_h5=read_from_h5py('features_simi_val.h5'),
        image_h5=read_from_h5py('features_protest.h5'),
        threshold=0.88
    )

    predict_via_feature(
        predict_h5=read_from_h5py('features_protest_val.h5'),
        image_h5=read_from_h5py('features_protest.h5'),
        threshold=0.88
    )

