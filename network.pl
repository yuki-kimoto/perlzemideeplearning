use strict;
use warnings;
use FindBin;

use JSON::PP;

# 学習率
my $learning_rate = 0.5;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 400;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 活性化関数
my $activate_func = \&sigmoid;

# 活性化関数の導関数
my $activate_func_derivative = \&sigmoid_derivative;

# 損失関数
my $cost_func = \&cross_entropy_cost;

# 損失関数の導関数
my $cost_func_derivative = \&cross_entropy_cost_derivative;

# 各層のニューロンの数
# 28 * 28 = 728のモノクロ画像を (入力層)
# 30個の中間出力を通って        (隠れ層)
# 0～9の10個に分類する          (出力層)
my $neurons_length_in_layers = [728, 30, 10];

# バイアスの初期化 - バイアスは各層の入力から出力への変換に利用されるので、バイアスの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。
# すべて0
my $biases_in_layers = [];
for (my $layer_index = 1; $layer_index < @$neurons_length_in_layers; $layer_index++) {
  my $neurons_length = $neurons_length_in_layers->[$layer_index];
  for (my $biase_index = 0; $biase_index < $neurons_length; $biase_index++) {
    $biases_in_layers->[$layer_index] ||= [];
    $biases_in_layers->[$layer_index][$biase_index] = 0;
  }
}

=pod
# 重みの初期化 - 重みは各層の入力から出力への変換に利用されるので、重みの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。
my $weights_in_layers = [];
for (my $layer_index = 0; $layer_index < @$neurons_length_in_layers - 1; $layer_index++) {
  my $neurons_length = $neurons_length_in_layers->[$layer_index];
  for (my $weight_index = 0; $weight_index < $neurons_length; $weight_index++) {
    $weights_in_layers->[$layer_index] ||= [];
    $weights_in_layers->[$layer_index][$weight_index] = 0;
  }
}
=cut

# MNIEST画像情報を読み込む - 入力用につかう手書きの訓練データ
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# MNIESTラベル情報を読み込む - 手書きの訓練データの期待される出力
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# シグモイド関数
sub sigmoid {
  my ($x) = @_;
  
  my $sigmoid = 1.0 / (1.0 + exp(-$x));
  
  return $sigmoid;
}

# シグモイド関数の導関数
sub sigmoid_derivative {
  my ($x) = @_;
  
  my $sigmoid_derivative = sigmoid($x) * (1 - sigmoid($x));
  
  return $sigmoid_derivative;
}

# クロスエントロピーコスト
sub cross_entropy_cost {
  my ($vec_a, $vec_y) = @_;
  
  my $fn = 0;
  for (my $i = 0; $i < @$vec_a; $i++) {
    $fn += -$vec_y * log($vec_a->[$i]) - (1 - $vec_y->[$i]) * log(1 - $vec_a->[$i]);
  }
  
  return $fn;
}

# クロスエントロピーコストの導関数
sub cross_entropy_cost_derivative {
  my ($vec_a, $vec_y) = @_;
  
  my $vec_out = [];
  for (my $i = 0; $i < @$vec_a; $i++) {
    $vec_out->[$i] = $vec_a->[$i] - $vec_y->[$i];
  }
  
  return $vec_out;
}

# MNIST画像情報を読み込む
sub load_mnist_train_image_file {
  my ($mnist_image_file) = @_;
  
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
  
  return $image_info;
}

# 正規分布に従う乱数を求める関数
# $sigma は標準偏差、$m は平均
sub randn {
  my ($m, $sigma) = @_;
  my ($r1, $r2) = (rand(), rand());
  while ($r1 == 0) { $r1 = rand(); }
  return ($sigma * sqrt(-2 * log($r1)) * sin(2 * 3.14159265359 * $r2)) + $m;
}

# MNIST画像情報を読み込む
sub load_mnist_train_label_file {
  my ($mnist_label_file) = @_;

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
}

=pod
sub init_weights {
  my $x = [0.5, 0.8];
  my $y = [0, 0, 0];
  my $x_len = @$x;
  my $y_len = @$y;

  my $w = [];
  for (my $i = 0; $i < $x_len * $y_len; $i++) {
    my $w_init_value = randn(0, sqrt(2/$x_len));
    push @$w, $w_init_value;
  }

  print STDERR Dumper($w);

  my $b = [
    0,
    0,
    0
  ];

  for (my $y_index = 0; $y_index < $y_len; $y_index++) {
    my $total = 0;
    for (my $x_index = 0; $x_index < $x_len; $x_index++) {
      $total += ($w->[$x_len * $y_index + $x_index] * $x->[$x_index]);
    }
    $total +=  $b->[$y_index];
    $y->[$y_index] = $total > 0 ? $total : 0;
  }

  print "($y->[0], $y->[1], $y->[2])\n";

  </pre>

  出力結果の例。

  <pre>
  $VAR1 = [
            '0.152110884289137',
            '2.40437412125725',
            '1.47474871999698',
            '0.30283258213298',
            '-0.274498796676187',
            '-1.27516026508991'
          ];
  (1.99955473915037, 0.979640425704874, 0)
}
=cut
