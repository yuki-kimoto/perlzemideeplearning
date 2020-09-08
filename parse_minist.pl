use strict;
use warnings;
use FindBin;

# MINIST画像情報を読み込む
my $minist_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";

open my $minist_image_fh, '<', $minist_image_file
  or die "Can't open file $minist_image_file: $!";

# マジックナンバー
my $image_buffer;
read($minist_image_fh, $image_buffer, 4);
my $magic_number = unpack('N1', $image_buffer);
if ($magic_number != 0x00000803) {
  die "Invalid magic number expected " . 0x00000803 . "actual $magic_number";
}

# 画像数
read($minist_image_fh, $image_buffer, 4);
my $items_count = unpack('N1', $image_buffer);

# 画像の行ピクセル数
read($minist_image_fh, $image_buffer, 4);
my $rows_count = unpack('N1', $image_buffer);

# 画像の列ピクセル数
read($minist_image_fh, $image_buffer, 4);
my $columns_count = unpack('N1', $image_buffer);

# 画像の読み込み
my $image_data;
my $all_images_length = $items_count * $rows_count * $columns_count;
my $read_length = read $minist_image_fh, $image_data, $all_images_length;
unless ($read_length == $all_images_length) {
  die "Can't read all images";
}

# 画像情報
my $image_info = {};
$image_info->{items_count} = $items_count;
$image_info->{rows_count} = $rows_count;
$image_info->{columns_count} = $columns_count;
$image_info->{data} = $image_data;

__END__

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