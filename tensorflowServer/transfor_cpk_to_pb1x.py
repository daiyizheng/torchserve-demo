# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 10:19
# software: PyCharm

"""
文件说明：
    tensorflow 1.x 版本
"""

import json
import os
import tensorflow as tf
import argparse

from tensorflowServer import modeling


def create_model(bert_config, is_training, input_ids):
    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids)
    output = model.get_pooled_output()
    # output = model.get_sequence_output()

    return output


def transfer_saved_model(args):
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = False # 是否使用GPU
    sess = tf.Session(config=gpu_config)

    print("going to restore checkpoint")
    bert_config_file = os.path.join(args.model_path, 'bert_config.json')
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    input_ids = tf.placeholder(tf.int32, [None, args.max_seq_len], name="input_ids")
    output = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_path+"/bert_model.ckpt")# tf.train.latest_checkpoint(args.model_path) 有 checkpoint 文件

    tf.saved_model.simple_save(sess, args.export_path, inputs={'input_ids': input_ids}, outputs={"outputs": output})
    print('savedModel export finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trans ckpt file to .pb file')

    parser.add_argument('-model_path', type=str, required=True, help='dir of a pretrained BERT model')
    parser.add_argument('-export_path', type=str, default=None, help='export model path')
    parser.add_argument('-max_seq_len', type=int, default=128, help='maximum length of a sequence')
    args = parser.parse_args()

    transfer_saved_model(args)


## python3 export_bert_test.py -model_path chinese_L-12_H-768_A-12  -export_path bert_output -max_seq_len 128