use strict;
use warnings;
use FindBin;

# MNISTラベル情報を読み込む
my $mnist_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";

open my $mnist_label_fh, '<', $mnist_label_file
  or die "Can't open file $mnist_label_file: $!";

# マジックナンバー
my $label_buffer;
read($mnist_label_fh, $label_buffer, 4);
my $magic_number = unpack('N1', $label_buffer);
if ($magic_number != 0x00000801) {
  die "Invalid magic number expected " . 0x00000801 . "actual $magic_number";
}

# ラベル数
read($mnist_label_fh, $label_buffer, 4);
my $items_count = unpack('N1', $label_buffer);

# ラベルの読み込み
my $label_numbers = [];
for (my $i = 0; $i < $items_count; $i++) {
  read $mnist_label_fh, $label_buffer, 1;
  my $label_number = unpack('C1', $label_buffer);
  push @$label_numbers, $label_number;
}

# ラベル情報
my $label_info = {};
$label_info->{items_count} = $items_count;
$label_info->{label_numbers} = $label_numbers;
