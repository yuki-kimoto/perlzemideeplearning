use strict;
use warnings;
use FindBin;
use Imager;

# MNIST画像情報を読み込む
my $mnist_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";

open my $mnist_image_fh, '<', $mnist_image_file
  or die "Can't open file $mnist_image_file: $!";

# マジックナンバー
my $image_buffer;
read($mnist_image_fh, $image_buffer, 4);
my $magic_number = unpack('N1', $image_buffer);
if ($magic_number != 0x00000803) {
  die "Invalid magic number expected " . 0x00000803 . "actual $magic_number";
}

# 画像数
read($mnist_image_fh, $image_buffer, 4);
my $items_count = unpack('N1', $image_buffer);

# 画像の行ピクセル数
read($mnist_image_fh, $image_buffer, 4);
my $rows_count = unpack('N1', $image_buffer);

# 画像の列ピクセル数
read($mnist_image_fh, $image_buffer, 4);
my $columns_count = unpack('N1', $image_buffer);

# 画像の読み込み
my $image_data;
my $all_images_length = $items_count * $rows_count * $columns_count;
my $read_length = read $mnist_image_fh, $image_data, $all_images_length;
unless ($read_length == $all_images_length) {
  die "Can't read all images";
}

# 画像情報
my $image_info = {};
$image_info->{items_count} = $items_count;
$image_info->{rows_count} = $rows_count;
$image_info->{columns_count} = $columns_count;
$image_info->{data} = $image_data;

# 画像情報の出力
for (my $i = 0; $i < 40000; $i++) {

  # 画像オフセット
  my $offset = $i * $rows_count * $columns_count;

  # キャンバス(モノクロ)
  my $img = Imager->new(xsize => $rows_count, ysize => $columns_count, channels => 1);
  
  # 画像情報を順番に出力
  for (my $row = 0; $row < $rows_count; $row++) {
    for (my $column = 0; $column < $columns_count; $column++) {
      
      # 色(白黒がRGBと逆なので反転)
      my $pos = $offset + ($column * $rows_count) + $row;
      my $color_bin = substr($image_data, $pos, 1);
      my $color_value = unpack('C1', $color_bin);
      my $color_value_neg = $color_value ^ 0xFF;
      my $color = Imager::Color->new($color_value_neg, $color_value_neg, $color_value_neg);
      
      # ピクセル描画
      $img->setpixel(x => $row, y => $column, color => $color);
    }
  }
  # Web表示できるようにPNGとして保存
  my $bitmap_file = "$FindBin::Bin/tmp_images/number$i.png";
  $img->write(file => $bitmap_file);
}

