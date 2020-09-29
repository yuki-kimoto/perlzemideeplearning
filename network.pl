use strict;
use warnings;
use FindBin;
use List::Util 'shuffle';

use JSON::PP;

# 学習率
my $learning_rate = 0.5;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 1;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 各層のニューロンの数
# 28 * 28 = 728のモノクロ画像を (入力層)
# 30個の中間出力を通って        (隠れ層)
# 0～9の10個に分類する          (出力層)
my $neurons_count_in_layers = [728, 30, 10];

# 各層のバイアス
my $biases_in_layers = [];

# 各層の重み
my $weights_mat_in_layers = [];

# 各層のバイアスと重みの初期化
for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
  my $input_neurons_count = $neurons_count_in_layers->[$layer_index];
  my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
  
  # バイアスの初期化 - バイアスは各層の入力から出力への変換に利用されるので、バイアスの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。
  # 0で初期化
  $biases_in_layers->[$layer_index] = array_new_zero($output_neurons_count);
  
  # 重みの初期化 - 重みは各層の入力から出力への変換に利用されるので、重みの組の数は、入力層、隠れ層、出力層の合計より1小さいことに注意。
  # 重みは列優先の行列と考える
  # Xivierの初期値で初期化
  my $weights_mat = mat_new_zero($output_neurons_count, $input_neurons_count);
  my $weights_length = $weights_mat->{rows_length} * $weights_mat->{columns_length};
  $weights_mat->{values} = array_create_xivier_init_value($input_neurons_count, $weights_length);
  
  $weights_mat_in_layers->[$layer_index] = $weights_mat;
}

# MNIEST画像情報を読み込む - 入力用につかう手書きの訓練データ
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# MNIESTラベル情報を読み込む - 手書きの訓練データの期待される出力
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# 訓練データのインデックス(最初の4万枚だけを訓練用データとして利用する。残りの1万枚は検証用データとする)
my @training_data_indexes = (0 .. 40000);

# エポックの回数だけ訓練セットを実行
for (my $epoch_index = 0; $epoch_index < $epoch_count; $epoch_index++) {
  
  # 訓練データのインデックスをシャッフル(ランダムに学習させた方が汎用化するらしい)
  my @training_data_indexes_shuffle = shuffle @training_data_indexes;
  
  my $count = 0;
  
  # ミニバッチサイズ単位で学習
  my $backprop_count = 0;
  while (my @indexed_for_mini_batch = splice(@training_data_indexes_shuffle, 0, $mini_batch_size)) {
    # ミニバッチにおけるバイアスの傾きの合計
    my $biase_grads_total_in_mini_batch = [];

    # ミニバッチにおける重みの傾きの合計
    my $weight_grads_total_in_mini_batch = [];
    
    for my $training_data_index (@indexed_for_mini_batch) {
      # バックプロパゲーションを使って重みとバイアスの損失関数に関する傾きを取得
      my $grads = backprop($neurons_count_in_layers, $mnist_train_image_info, $mnist_train_label_info, $training_data_index);
      
      # バイアスの損失関数に関する傾き
      my $biase_grads = $grads->{biase};
      
      # 重みの損失関数に関する傾き
      my $weight_grads = $grads->{weight};
      
      # 各層のバイアスを更新(学習率を考慮し、ミニバッチ数で割る)
      for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
        my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
        for (my $biase_index = 0; $biase_index < $output_neurons_count; $biase_index++) {
          $biases_in_layers->[$layer_index][$biase_index] -= ($learning_rate / $mini_batch_size) * $biase_grads->[$layer_index][$biase_index];
        }
      }
      
      # 各層の重みを更新(学習率を考慮し、傾きの合計をミニバッチ数で、ミニバッチ数で割る)
      for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
        my $input_neurons_count = $neurons_count_in_layers->[$layer_index];
        my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
        my $weights_length = $input_neurons_count * $output_neurons_count;
        for (my $weight_index = 0; $weight_index < $weights_length; $weight_index++) {
          $weights_mat_in_layers->[$layer_index]{values}[$weight_index] -= ($learning_rate / $mini_batch_size) * $weight_grads->[$layer_index][$weight_index];
        }
      }
    }
  }
}

# バックプロパゲーション
sub backprop {
  my ($neurons_count_in_layers, $mnist_train_image_info, $mnist_train_label_info, $training_data_index) = @_;
  
  # 入力
  my $image_unit_length = $mnist_train_image_info->{rows_count} *  $mnist_train_image_info->{columns_count};
  my $mnist_train_image_data = $mnist_train_image_info->{data};
  my $first_inputs_packed = substr($mnist_train_image_data, $image_unit_length * $training_data_index, $image_unit_length);
  my $first_inputs = [unpack("C$image_unit_length", $first_inputs_packed)];
  
  # 期待される出力(確率化する)
  my $label_number = $mnist_train_label_info->{label_numbers}[$training_data_index];
  my $desired_outputs = probabilize_outputs($label_number);
  
  # バイアスの傾きを0で初期化
  my $biase_grads_in_layers = [];
  for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
    my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
    $biase_grads_in_layers->[$layer_index] = [(0) x $output_neurons_count];
  }

  # 重みの傾きを0で初期化
  my $weight_grads_in_layers = [];
  for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
    my $input_neurons_count = $neurons_count_in_layers->[$layer_index];
    my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
    my $weights_length = $input_neurons_count * $output_neurons_count;
    $weight_grads_in_layers->[$layer_index] = [(0) x $weights_length];
  }

  # 各層の入力
  my $inputs_in_layers = [$first_inputs];
  
  # 各層の活性化された出力
  my $outputs_in_layers = [];
  
  # 入力層の入力から出力層の出力を求める
  # バックプロパゲーションのために各層の出力と活性化された出力を保存
  for (my $layer_index = 0; $layer_index < @$neurons_count_in_layers - 1; $layer_index++) {
    my $cur_inputs = $inputs_in_layers->[-1];
    my $input_neurons_count = $neurons_count_in_layers->[$layer_index];
    my $output_neurons_count = $neurons_count_in_layers->[$layer_index + 1];
    
    # 重み行列
    my $weights_mat = $weights_mat_in_layers->[$layer_index];
    
    # 入力行列
    my $cur_inputs_rows_length = $output_neurons_count;
    my $cur_inputs_columns_length = 1;
    my $cur_inputs_mat = {
      rows_length => $cur_inputs_rows_length,
      columns_length => $cur_inputs_columns_length,
      values => $cur_inputs,
    };
    
    # 重みと入力の行列積
    my $mul_weights_inputs_mat = mat_mul($weights_mat, $cur_inputs_mat);
    my $mul_weights_inputs = $mul_weights_inputs_mat->{values};
    
    # バイアス
    my $biases = $biases_in_layers->[$layer_index];
    
    # 出力 - 重みと入力の行列積とバイアスの和
    my $outputs = array_add($mul_weights_inputs, $biases);
    
    # 活性化された出力 - 出力に活性化関数を適用
    my $activate_outputs = array_sigmoid($outputs);
    
    # バックプロパゲーションのために出力を保存
    push @$outputs_in_layers, $outputs;
    
    # 現在の入力を更新
    $cur_inputs = $activate_outputs;
    
    # バックプロパゲーションのために次の入力を保存
    push @$inputs_in_layers, $activate_outputs;
  }
  
  # 最後の出力
  my $last_outputs = $outputs_in_layers->[-1];
  
  # 最後の活性化された出力
  my $last_activate_outputs = pop @$inputs_in_layers;

  # 誤差
  my $cost = cross_entropy_cost($last_activate_outputs, $desired_outputs);
  
  print "Cost: $cost\n";
  
  # 活性化された出力の微小変化 / 最後の出力の微小変化 
  my $grads_last_outputs_to_activate_func = array_sigmoid_derivative($last_outputs);
  
  # 損失関数の微小変化 / 最後に活性化された出力の微小変化
  my $grads_last_activate_outputs_to_cost_func = cross_entropy_cost_derivative($last_activate_outputs, $desired_outputs);
  
  # 損失関数の微小変化 / 最後の出力の微小変化 (合成微分)
  my $grads_last_outputs_to_cost_func = [];
  for (my $i = 0; $i < @$grads_last_outputs_to_activate_func; $i++) {
    $grads_last_outputs_to_cost_func->[$i] = $grads_last_outputs_to_activate_func->[$i] * $grads_last_activate_outputs_to_cost_func->[$i];
  }
  
  # 損失関数の微小変化 / 最終の層のバイアスの微小変化
  my $last_biase_grads = $grads_last_outputs_to_cost_func;
  
  # 損失関数の微小変化 / 最終の層の重みの微小変化
  my $last_weight_grads = [];
  my $last_inputs = $inputs_in_layers->[-1];
  for (my $last_inputs_index = 0; $last_inputs_index < @$last_inputs; $last_inputs_index++) {
    for (my $last_biase_grads_index = 0; $last_biase_grads_index < @$last_biase_grads; $last_biase_grads_index++) {
      $last_weight_grads->[$last_biase_grads_index + @$last_biase_grads * $last_inputs_index]
        = $last_biase_grads->[$last_biase_grads_index] * $last_inputs->[$last_inputs_index];
    }
  }
  
  $biase_grads_in_layers->[@$biase_grads_in_layers - 1] = $last_biase_grads;
  $weight_grads_in_layers->[@$biase_grads_in_layers - 1] = $last_weight_grads;
  
  # 最後の重みとバイアスの変換より一つ前から始める
  for (my $layer_index = @$neurons_count_in_layers - 3; $layer_index >= 0; $layer_index--) {
    # 活性化された出力の微小変化 / 出力の微小変化
    my $outputs = $outputs_in_layers->[$layer_index];

    # 損失関数の微小変化 / この層のバイアスの微小変化(バックプロパゲーションで求める)
    # 次の層の重みの傾きの転置行列とバイアスの傾きの転置行列をかけて、それぞれの要素に、活性化関数の導関数をかける
    my $biase_grads = [];
    my $forword_biase_grads = $biase_grads_in_layers->[$layer_index + 1];
    my $forword_weight_grads = $weight_grads_in_layers->[$layer_index + 1];
    my $forword_weight_columns_length = @$forword_weight_grads / @$forword_biase_grads;
    for (my $biase_index = 0; $biase_index < @$forword_biase_grads; $biase_index++) {
      for (my $weight_columns_index = 0; $weight_columns_index < $forword_weight_columns_length; $weight_columns_index++) {
        $biase_grads->[$weight_columns_index] += $forword_biase_grads->[$biase_index] * $forword_weight_grads->[$biase_index + @$forword_biase_grads * $weight_columns_index];
      }
    }
    
    $biase_grads = array_sigmoid_derivative($biase_grads);
    
    # 損失関数の微小変化 / この層の重みの微小変化(バックプロパゲーションで求める)
    my $weights_grads = [];
    my $inputs = $inputs_in_layers->[$layer_index];
    for (my $inputs_index = 0; $inputs_index < @$inputs; $inputs_index++) {
      for (my $biase_grads_index = 0; $biase_grads_index < @$biase_grads; $biase_grads_index++) {
        $weights_grads->[$biase_grads_index + @$biase_grads * $inputs_index]
          = $biase_grads->[$biase_grads_index] * $inputs->[$inputs_index];
      }
    }
    $weight_grads_in_layers->[$layer_index] = $weights_grads;
  }

  my $grads = {};
  $grads->{biase} = $biase_grads_in_layers;
  $grads->{weight} = $weight_grads_in_layers;
  
  return $grads;
}

sub probabilize_outputs {
  my ($label_number) = @_;
  
  my $desired_outputs = [];
  for (my $desired_outputs_index = 0; $desired_outputs_index < 10; $desired_outputs_index++) {
    if ($label_number == $desired_outputs_index) {
      $desired_outputs->[$desired_outputs_index] = 1;
    }
    else {
      $desired_outputs->[$desired_outputs_index] = 0;
    }
  }
  
  return $desired_outputs;
}

# 配列の各要素の和
sub array_add {
  my ($nums1, $nums2) = @_;
  
  if (@$nums1 != @$nums2) {
    die "Array length is diffent";
  }
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums1; $i++) {
    $nums_out->[$i] = $nums1->[$i] + $nums2->[$i];
  }
  
  return $nums_out;
}

# 配列の各要素の積
sub array_mul {
  my ($nums1, $nums2) = @_;
  
  if (@$nums1 != @$nums2) {
    die "Array length is diffent";
  }
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums1; $i++) {
    $nums_out->[$i] = $nums1->[$i] * $nums2->[$i];
  }
  
  return $nums_out;
}

# Xivierの初期値を取得
sub create_xivier_init_value {
  my ($input_neurons_count) = @_;
  
  return randn(0, 1 / sqrt($input_neurons_count));
}

# 配列の各要素にXivierの初期値を取得を適用する
sub array_create_xivier_init_value {
  my ($input_neurons_count, $length) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < $length; $i++) {
    $nums_out->[$i] = create_xivier_init_value($input_neurons_count);
  }
  
  return $nums_out;
}

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

# 配列の各要素にシグモイド関数を適用する
sub array_sigmoid {
  my ($nums) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums; $i++) {
    $nums_out->[$i] = sigmoid($nums->[$i]);
  }
  
  return $nums_out;
}

# 配列の各要素にシグモイド関数の導関数を適用する
sub array_sigmoid_derivative {
  my ($nums) = @_;
  
  my $nums_out = [];
  for (my $i = 0; $i < @$nums; $i++) {
    $nums_out->[$i] = sigmoid_derivative($nums->[$i]);
  }
  
  return $nums_out;
}

# クロスエントロピーコスト
sub cross_entropy_cost {
  my ($vec_a, $vec_y) = @_;
  
  my $cross_entropy_cost = 0;
  for (my $i = 0; $i < @$vec_a; $i++) {
    my $tmp = -$vec_y->[$i] * log($vec_a->[$i]) - (1 - $vec_y->[$i]) * log(1 - $vec_a->[$i]);
    $cross_entropy_cost += $tmp;
  }

  return $cross_entropy_cost;
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

# 正規分布に従う乱数を求める関数
# $m は平均, $sigma は標準偏差、
sub randn {
  my ($m, $sigma) = @_;
  my ($r1, $r2) = (rand(), rand());
  while ($r1 == 0) { $r1 = rand(); }
  return ($sigma * sqrt(-2 * log($r1)) * sin(2 * 3.14159265359 * $r2)) + $m;
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
  
  return $label_info;
}

# 配列を0で初期化
sub array_new_zero {
  my ($length) = @_;
  
  my $nums = [(0) x $length];
  
  return $nums;
}

# 行列を0で初期化
sub mat_new_zero {
  my ($rows_length, $columns_length) = @_;
  
  my $values_length = $rows_length * $columns_length;
  my $mat = {
    rows_length => $rows_length,
    columns_length => $columns_length,
    values => [(0) x $values_length],
  };
  
  return $mat;
}

sub mat_mul {
  my ($mat1, $mat2) = @_;
  
  my $mat1_rows_length = $mat1->{rows_length};
  my $mat1_columns_length = $mat1->{columns_length};
  my $mat1_values = $mat1->{values};
  
  my $mat2_rows_length = $mat2->{rows_length};
  my $mat2_columns_length = $mat2->{columns_length};
  my $mat2_values = $mat2->{values};
  
  # 行列の積の計算
  my $mat_out_values = [];
  for(my $row = 0; $row < $mat1_rows_length; $row++) {
    for(my $col = 0; $col < $mat2_columns_length; $col++) {
      for(my $incol = 0; $incol < $mat1_columns_length; $incol++) {
        $mat_out_values->[$row + $col * $mat2_rows_length]
         += $mat1_values->[$row + $incol * $mat1_rows_length] * $mat2_values->[$incol + $col * $mat2_rows_length];
      }
    }
  }
  
  my $mat_out = {
    rows_length => $mat1_rows_length,
    columns_length => $mat2_columns_length,
    values => $mat_out_values,
  };
  
  return $mat_out;
}

# 列優先の行列の作成
sub mat_new {
  my ($values, $rows_length, $columns_length) = @_;
  
  my $mat = {
    rows_length => $rows_length,
    columns_length => $columns_length,
    values => $values,
  };
  
  return $mat;
}
