import os
import pickle
import sys

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from os import path

# https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
from utils.util import get_ids_dict

sys.path.append(path.abspath('/home/natasha/PycharmProjects/pytorch-yolo-v3/'))
from util import write_results


def delta(coord, bad_values_counter):
    if not 0.0 <= coord <= 1.0:
        if coord < 0.0:
            bad_values_counter = bad_values_counter + np.abs(coord)
            coord = 0.0
        if coord > 1.0:
            bad_values_counter = bad_values_counter + np.abs(coord - 1.0)
            coord = 1.0
    return bad_values_counter, coord


# For each image in the test set,
# you must predict a list of boxes describing objects in the image.
# Each box is described as

# <label confidence x_min y_min x_max y_max>.

# The length of your PredictionString should always be multiple of 6.
# If there is no boxes predicted for a given image, PredictionString should be empty.
# Every value is space delimited.

# /m/05s2s 0.9 0.46 0.08 0.93 0.5 /m/0c9ph5 0.5 0.25 0.6 0.6 0.9

# outputs structure is as follows:
# it is a tensor of shape total_number_of_predicted_boxes_for_tis_batch x 8
# each row contains
# index of the image in the current batch, box coordinates, objectness score, class score, class number
# example:
# [2.0000, 210.5068, 90.1572, 227.4409, 129.0678, 0.7430, 0.9925, 56.0000]
def get_part_of_prediction_string_from_output(output, ids_dict, bad_values_counter):
    output = output.cpu().numpy()
    size = 320.0
    label_name = ids_dict[int(output[7])]
    confidence = output[5]
    x_min = output[1] / size
    y_min = output[2] / size
    x_max = output[3] / size
    y_max = output[4] / size

    bad_values_counter, x_min = delta(x_min, bad_values_counter)
    bad_values_counter, y_min = delta(y_min, bad_values_counter)
    bad_values_counter, x_max = delta(x_max, bad_values_counter)
    bad_values_counter, y_max = delta(y_max, bad_values_counter)
    # print('bad_values_counter', bad_values_counter)
    # assert 0.0 <= x_min <= 1.0, 'For %s x_min = %f' % (output, x_min)
    # assert 0.0 <= y_min <= 1.0, 'For %s y_min = %f' % (output, y_min)
    # assert 0.0 <= x_max <= 1.0, 'For %s x_max = %f' % (output, x_max)
    # assert 0.0 <= y_max <= 1.0, 'For %s y_max = %f' % (output, y_max)
    return label_name + ' ' + str(confidence) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(
        y_max) + ' ', bad_values_counter


def outputs_for_large_dataset(loader, network):
    config = loader.config
    torch.cuda.empty_cache()
    name = loader.dataset.name
    # batches_number = save_inference_results_on_disk(loader, network, name)
    batches_number = 625
    name = 'test'
    return read_inference_results_from_disk(config, batches_number, name)


def read_inference_results_from_disk(config, batches_number, name):
    path = os.path.join(config['temp_folder'], name, '')
    pack_volume = config['pack_volume']
    assert 'all_outputs_%d' % pack_volume in os.listdir(path), \
        'There should be precomputed inference data in %s!' % path

    all_outputs = torch.cuda.FloatTensor()
    for i in range(1, batches_number + 1):
        outputs = torch.load('%sall_outputs_%d' % (path, i * pack_volume))
        all_outputs = torch.cat((all_outputs, outputs), dim=0)
    with open('%sall_ids.pkl' % path, 'rb') as f:
        all_ids = pickle.load(f)
    with open('%sall_empty_ids.pkl' % path, 'rb') as f:
        all_empty_ids = pickle.load(f)

    return all_outputs, all_ids, all_empty_ids


def save_inference_results_on_disk(loader, network, name):
    config = loader.config
    pack_volume = config['pack_volume']
    path = os.path.join(config['temp_folder'], name, '')
    print('path ', path)
    network.eval()

    all_outputs = torch.cuda.FloatTensor()
    all_ids = []
    all_empty_ids = []
    i = 1
    print('Inference is in progress')
    print('loader ', loader.batch_sampler.sampler)
    for data in tqdm(loader):
        images_tensors, target_vectors, images_ids = data
        outputs = network(Variable(images_tensors).cuda())
        # confidence here is a threshold confidence

        # outputs structure is as follows:
        # it is a tensor of shape total_number_of_predicted_boxes_for_tis_batch x 8
        # each row contains
        # index of the image in the current batch, box coordinates, objectness score, class score, class number
        # example:
        # [2.0000, 210.5068, 90.1572, 227.4409, 129.0678, 0.7430, 0.9925, 56.0000]
        outputs = write_results(outputs, confidence=0.5, num_classes=80, nms=True, nms_conf=0.4)

        if not isinstance(outputs, int):
            images_with_predictions_ids_in_the_current_batch = outputs[:, 0].int()
            images_without_predictions_ids_in_the_current_batch = np.delete(
                np.arange(len(images_ids)),
                images_with_predictions_ids_in_the_current_batch
            )
            for empty_id in images_without_predictions_ids_in_the_current_batch:
                all_empty_ids.append(images_ids[empty_id])
            all_outputs = torch.cat((all_outputs, outputs.data), dim=0)
            for id in images_with_predictions_ids_in_the_current_batch:
                all_ids.append(images_ids[id])
            if i % pack_volume == 0:
                torch.save(all_outputs, '%sall_outputs_%d' % (path, i))
                all_outputs = torch.cuda.FloatTensor()
                torch.cuda.empty_cache()
        i += 1
    batches_number = len(loader) // pack_volume
    print('batches_number = ', batches_number)
    with open('%sall_ids.pkl' % path, 'wb') as f:
        pickle.dump(all_ids, f)

    with open('%sall_empty_ids.pkl' % path, 'wb') as f:
        pickle.dump(all_empty_ids, f)
    all_outputs = None
    torch.cuda.empty_cache()
    return batches_number


def inference(loader, model):
    all_outputs, all_ids, all_empty_ids = outputs_for_large_dataset(loader, model)
    print('all_ids ', all_ids[-5:-1])
    print('len(all_ids)', len(all_ids))
    ids_dict = get_ids_dict()

    prediction_strings = []
    ids = []
    id_previous = all_ids[0]
    prediction_string = ''
    bad_values_counter = 0
    for id, output in zip(all_ids, all_outputs):
        print('id ', id)
        new_part_of_prediction_string, bad_values_counter = get_part_of_prediction_string_from_output(output, ids_dict,
                                                                                                      bad_values_counter)
        if id == id_previous:
            prediction_string = prediction_string + new_part_of_prediction_string
        else:
            prediction_strings.append(prediction_string.strip())
            ids.append(id_previous)
            prediction_string = new_part_of_prediction_string
            id_previous = id
    print('bad_values_counter ', bad_values_counter)
    print('prediction_string ', prediction_string)
    prediction_strings.append(prediction_string.strip())
    ids.append(id)
    print('ids ', ids[-5:])

    print('len(all_empty_ids) ', len(all_empty_ids))
    for empty_id in all_empty_ids:
        ids.append(empty_id)
        prediction_strings.append('')

    print('len(ids)', len(ids))
    print('len(prediction_strings)', len(prediction_strings))
    rows = []
    with open('natasha_submission.csv', 'w') as csv_file:
        csv_file.write('ImageId,PredictionString\n')
        for (id, prediction_string) in zip(ids, prediction_strings):
            row = str(id) + ',' + prediction_string + '\n'
            # print(row)
            # csv_file.write(row)
            rows.append(row)
        rows[-1] = rows[-1].replace('\n', '')
        csv_file.writelines(rows)
