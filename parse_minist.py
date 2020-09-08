import os
import struct

dataset_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'dataset')


def load_mnist_dataset():
    """This implementation is just for my training.
    You should use other popular libraries to load mnist dataset."""

    train_images_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')

    def load_images_file(path):
        with open(path, 'rb') as f:
            # check magic number
            magic_number = struct.unpack('>i', f.read(4))[0]
            if magic_number != 0x00000803:
                raise Exception("Invalid magic number: expected={}, actual={}"
                                .format(0x00000803, magic_number))

            # load number of items
            items_num = struct.unpack('>i', f.read(4))[0]

            # load numbers of rows and cols
            rows_num = struct.unpack('>i', f.read(4))[0]
            cols_num = struct.unpack('>i', f.read(4))[0]

            # load images
            images = [[[struct.unpack('B', f.read(1))[0]
                        for k in range(cols_num)]
                       for j in range(rows_num)]
                      for i in range(items_num)]

        return images

    def load_labels_file(path):
        with open(path, 'rb') as f:
            # check magic number
            magic_number = struct.unpack('>i', f.read(4))[0]
            if magic_number != 0x00000801:
                raise Exception("Invalid magic number: expected={}, actual={}"
                                .format(0x00000801, magic_number))

            # load number of items
            items_num = struct.unpack('>i', f.read(4))[0]

            # load labels
            labels = [struct.unpack('B', f.read(1))[0]
                      for i in range(items_num)]

        return labels

    train_images = load_images_file(train_images_path)
    train_labels = load_labels_file(train_labels_path)
    test_images = load_images_file(test_images_path)
    test_labels = load_labels_file(test_labels_path)

    return (train_images, train_labels, test_images, test_labels)