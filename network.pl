use strict;
use warnings;

use JSON::PP;

# 学習率
my $learning_rate = 0.5;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 400;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 活性化関数
my $activate_func = &sigmoid;

# 活性化関数の導関数
my $activate_func_derivative = &sigmoid_derivative;

# 損失関数
my $cost_func = &cross_entropy_cost;

# 損失関数の導関数
my $cost_func_derivative = &cross_entropy_cost_derivative;

# 各層のニューロンの数
# 28 * 28 = 728のモノクロ画像を (入力層)
# 30個の中間出力を通って        (隠れ層)
# 0～9の10個に分類する          (出力層)
my $neurons_length_in_layers = [728, 30, 10];

# 重みの初期化 - 重みは各層の入力から出力への変換に利用されるので、重みの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。

# バイアスの初期化 - バイアスは各層の入力から出力への変換に利用されるので、バイアスの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。

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
