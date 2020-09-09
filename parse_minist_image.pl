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
