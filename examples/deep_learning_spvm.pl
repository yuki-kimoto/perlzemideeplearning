use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/network_lib";
use List::Util 'shuffle';

use SPVM 'MyAIUtil';

my $api = SPVM::api();

# 学習率
my $learning_rate = 3;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 30;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 各層のニューロンの数
# 28 * 28 = 784のモノクロ画像を (入力)
# 30個の中間出力を通って        (中間出力)
# 0～9の10個に分類する          (出力)
my $neurons_count_in_layers = [784, 30, 10];

# 各層のm個の入力をn個の出力に変換する関数の情報。入力数、出力数、バイアス、重み
my $m_to_n_func_infos = [];

# 各層のニューロン数からmからnへの変換関数の情報を作成
for (my $i = 0; $i < @$neurons_count_in_layers - 1; $i++) {
  my $inputs_length = $neurons_count_in_layers->[$i];
  my $outputs_length = $neurons_count_in_layers->[$i + 1];
  
  # バイアスを0で初期化
  my $biases = SPVM::MyAIUtil->array_new_zero($outputs_length);

  # Xivierの初期値で重みを初期化。重みは列優先行列
  my $weights_mat = SPVM::MyAIUtil->mat_new_zero($outputs_length, $inputs_length);
  my $weights_length = $weights_mat->rows_length * $weights_mat->columns_length;
  $weights_mat->set_values(SPVM::MyAIUtil->array_create_xavier_init_value($weights_length, $inputs_length));
  
  # 変換関数の情報を設定
  $m_to_n_func_infos->[$i] = {
    inputs_length => $inputs_length,
    outputs_length => $outputs_length,
    biases => $biases,
    weights_mat => $weights_mat,
  };
}

# MNIEST画像情報を読み込む - 入力用につかう手書きの訓練データ
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# MNIESTラベル情報を読み込む - 手書きの訓練データの期待される出力
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# 訓練データのインデックス(最初の4万枚だけを訓練用データとして利用する。残りの1万枚は検証用データとする)
my @training_data_indexes = (0 .. 40000);

# ミニバッチ単位における各変換関数の情報
my $m_to_n_func_mini_batch_infos = [];

# ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
# メモリ領域を繰り返しつかうためここで初期化
for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
  my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
  my $biases = $m_to_n_func_info->{biases};
  my $weights_mat = $m_to_n_func_info->{weights_mat};
  
  # バイアスの長さ
  my $biases_length = $biases->length;
  
  # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
  my $biase_grad_totals = SPVM::MyAIUtil->array_new_zero($biases_length);
  $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals} = $biase_grad_totals;

  # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
  my $weight_grad_totals_mat = SPVM::MyAIUtil->mat_new_zero($weights_mat->rows_length, $weights_mat->columns_length);
  $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat} = $weight_grad_totals_mat;
}

# 総実行回数
my $total_count = 0;

# 正解数
my $answer_match_count = 0;

# エポックの回数だけ訓練セットを実行
for (my $epoch_index = 0; $epoch_index < $epoch_count; $epoch_index++) {
  
  # 訓練データのインデックスをシャッフル(ランダムに学習させた方が汎用化するらしい)
  my @training_data_indexes_shuffle = shuffle @training_data_indexes;
  
  my $count = 0;
  
  # ミニバッチサイズ単位で学習
  my $backprop_count = 0;
  
  warn "Epoch $epoch_index";
  
  while (my @indexed_for_mini_batch = splice(@training_data_indexes_shuffle, 0, $mini_batch_size)) {
    
    # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化
    for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_mini_batch_infos; $m_to_n_func_index++) {
      my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
      my $biases = $m_to_n_func_info->{biases};
      my $weights_mat = $m_to_n_func_info->{weights_mat};
      
      # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals} = SPVM::MyAIUtil->array_new_zero($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals}->length);
      
      # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}->set_values(SPVM::MyAIUtil->array_new_zero($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}->values->length));
    }
    
    for my $training_data_index (@indexed_for_mini_batch) {
      # バックプロパゲーションを使って重みとバイアスの損失関数に関する傾きを取得
      my $m_to_n_func_grad_infos = backprop($m_to_n_func_infos, $mnist_train_image_info, $mnist_train_label_info, $training_data_index);
      
      # バイアスの損失関数に関する傾き
      my $biase_grads = $m_to_n_func_grad_infos->{biases};
      
      # 重みの損失関数に関する傾き
      my $weight_grads_mat = $m_to_n_func_grad_infos->{weights_mat};

      # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きを加算
      for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_mini_batch_infos; $m_to_n_func_index++) {
        my $m_to_n_func_info = $m_to_n_func_infos->[$m_to_n_func_index];
        
        # ミニバッチにおける各変換関数のバイアスの傾きを加算
        SPVM::MyAIUtil->array_add_inplace($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals}, $biase_grads->[$m_to_n_func_index]);

        # ミニバッチにおける各変換関数の重みの傾きを加算
        SPVM::MyAIUtil->array_add_inplace($m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}->values, $weight_grads_mat->[$m_to_n_func_index]->values);
      }
    }

    # 各変換関数のバイアスと重みをミニバッチの傾きの合計を使って更新
    for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
      
      # 各変換関数のバイアスを更新(学習率を考慮し、ミニバッチ数で割る)
      SPVM::MyAIUtil->update_params($m_to_n_func_infos->[$m_to_n_func_index]{biases}, $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{biase_grad_totals}, $learning_rate, $mini_batch_size);
      
      # 各変換関数の重みを更新(学習率を考慮し、傾きの合計をミニバッチ数で、ミニバッチ数で割る)
      SPVM::MyAIUtil->update_params($m_to_n_func_infos->[$m_to_n_func_index]{weights_mat}->values, $m_to_n_func_mini_batch_infos->[$m_to_n_func_index]{weight_grad_totals_mat}->values, $learning_rate, $mini_batch_size);
    }
  }
}

# バックプロパゲーション
sub backprop {
  my ($m_to_n_func_infos, $mnist_train_image_info, $mnist_train_label_info, $training_data_index) = @_;
  
  my $first_inputs_length = $m_to_n_func_infos->[0]{inputs_length};
  
  # 入力(0～255の値を0～1に変換)
  my $image_unit_length = $mnist_train_image_info->{rows_count} *  $mnist_train_image_info->{columns_count};
  my $mnist_train_image_data = $mnist_train_image_info->{data};
  my $first_inputs_packed = substr($mnist_train_image_data, $image_unit_length * $training_data_index, $image_unit_length);
  my $first_inputs_raw_uint8 = [unpack("C$first_inputs_length", $first_inputs_packed)];
  my $first_inputs_raw_float = $api->new_float_array($first_inputs_raw_uint8);
  my $first_inputs = SPVM::MyAIUtil->array_div_scalar($first_inputs_raw_float, 255);
  
  # 期待される出力を確率分布化する
  my $label_number = $mnist_train_label_info->{label_numbers}[$training_data_index];
  my $desired_outputs = SPVM::MyAIUtil->probabilize_desired_outputs($label_number);
  
  # 各変換関数のバイアスの傾き
  my $biase_grads_in_m_to_n_funcs = [];
  
  # 各変換関数の重みの傾き
  my $weight_grads_mat_in_m_to_n_funcs = [];
  
  # バイアスの傾きと重みの傾きの初期化
  for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
    my $inputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{inputs_length};
    my $outputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{outputs_length};

    # バイアスの傾きを0で初期化
    $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index] = SPVM::MyAIUtil->array_new_zero($outputs_length);

    # 重みの傾きを0で初期化
    $weight_grads_mat_in_m_to_n_funcs->[$m_to_n_func_index] = SPVM::MyAIUtil->mat_new_zero($outputs_length, $inputs_length);
  }

  # 各層の入力
  my $inputs_in_m_to_n_funcs = [$first_inputs];
  
  # 各層の活性化された出力
  my $outputs_in_m_to_n_funcs = [];
  
  # 入力層の入力から出力層の出力を求める
  # バックプロパゲーションのために各層の出力と活性化された出力を保存
  for (my $m_to_n_func_index = 0; $m_to_n_func_index < @$m_to_n_func_infos; $m_to_n_func_index++) {
    my $cur_inputs = $inputs_in_m_to_n_funcs->[-1];
    my $inputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{inputs_length};
    my $outputs_length = $m_to_n_func_infos->[$m_to_n_func_index]{outputs_length};
    
    # 重み行列
    my $weights_mat = $m_to_n_func_infos->[$m_to_n_func_index]{weights_mat};

    # 入力行列
    my $cur_inputs_rows_length = $outputs_length;
    my $cur_inputs_columns_length = 1;
    my $cur_inputs_mat = SPVM::MyAIUtil->mat_new($cur_inputs, $cur_inputs_rows_length, $cur_inputs_columns_length);
    
    # 重みと入力の行列積
    my $mul_weights_inputs_mat = SPVM::MyAIUtil->mat_mul($weights_mat, $cur_inputs_mat);
    my $mul_weights_inputs = $mul_weights_inputs_mat->values;

    # バイアス
    my $biases = $m_to_n_func_infos->[$m_to_n_func_index]{biases};
    
    # 出力 - 重みと入力の行列積とバイアスの和
    my $outputs = SPVM::MyAIUtil->array_add($mul_weights_inputs, $biases);
    
    
    # 活性化された出力 - 出力に活性化関数を適用
    my $activate_outputs = SPVM::MyAIUtil->array_sigmoid($outputs);

    # バックプロパゲーションのために出力を保存
    push @$outputs_in_m_to_n_funcs, $outputs;
    
    # バックプロパゲーションのために次の入力を保存
    push @$inputs_in_m_to_n_funcs, $activate_outputs;
  }
  
  # 最後の出力
  my $last_outputs = $outputs_in_m_to_n_funcs->[-1];
  
  # 最後の活性化された出力
  my $last_activate_outputs = pop @$inputs_in_m_to_n_funcs;
  
  # softmax関数
  # my $softmax_outputs = SPVM::MyAIUtil->softmax($last_activate_outputs);
  
  # 誤差
  
  my $cost = SPVM::MyAIUtil->cross_entropy_cost($last_activate_outputs, $desired_outputs);
  # my $cost = SPVM::MyAIUtil->softmax_cross_entropy_cost($softmax_outputs, $desired_outputs);
  print "Cost: " . sprintf("%.3f", $cost) . "\n";
  
  # 正解したかどうか
  my $answer = SPVM::MyAIUtil->max_index($last_activate_outputs);
  # my $answer = SPVM::MyAIUtil->max_index($softmax_outputs);
  my $desired_answer = SPVM::MyAIUtil->max_index($desired_outputs);
  $total_count++;
  if ($answer == $desired_answer) {
    $answer_match_count++;
  }
  
  # 正解率を出力
  my $match_rate = $answer_match_count / $total_count;
  print "Match Rate: " . sprintf("%.02f", 100 * $match_rate) . "%\n";
  
  # 活性化された出力の微小変化 / 最後の出力の微小変化 
  my $grad_last_outputs_to_activate_func = SPVM::MyAIUtil->array_sigmoid_derivative($last_outputs);
  
  # 損失関数の微小変化 / 最後に活性化された出力の微小変化
  # my $grad_last_activate_outputs_to_cost_func = softmax_cross_entropy_cost_derivative($last_activate_outputs, $desired_outputs);
  my $grad_last_activate_outputs_to_cost_func = SPVM::MyAIUtil->cross_entropy_cost_derivative($last_activate_outputs, $desired_outputs);

  # 損失関数の微小変化 / 最後の出力の微小変化 (合成微分)
  my $grad_last_outputs_to_cost_func = SPVM::MyAIUtil->array_mul($grad_last_outputs_to_activate_func, $grad_last_activate_outputs_to_cost_func);

  # 損失関数の微小変化 / 最終の層のバイアスの微小変化
  my $last_biase_grads = $grad_last_outputs_to_cost_func;
  

  # 損失関数の微小変化 / 最終の層の重みの微小変化
  my $last_biase_grads_mat = SPVM::MyAIUtil->mat_new($last_biase_grads, $last_biase_grads->length, 1);
  my $last_inputs = $inputs_in_m_to_n_funcs->[@$inputs_in_m_to_n_funcs - 1];
  my $last_inputs_transpose_mat = SPVM::MyAIUtil->mat_new($last_inputs, 1, $last_inputs->length);
  my $last_weight_grads_mat = SPVM::MyAIUtil->mat_mul($last_biase_grads_mat, $last_inputs_transpose_mat);
    
  $biase_grads_in_m_to_n_funcs->[@$biase_grads_in_m_to_n_funcs - 1] = $last_biase_grads;
  $weight_grads_mat_in_m_to_n_funcs->[@$biase_grads_in_m_to_n_funcs - 1] = $last_weight_grads_mat;

        
  # 最後の重みとバイアスの変換より一つ前から始める
  for (my $m_to_n_func_index = @$m_to_n_func_infos - 2; $m_to_n_func_index >= 0; $m_to_n_func_index--) {
    # 活性化された出力の微小変化 / 出力の微小変化
    my $outputs = $outputs_in_m_to_n_funcs->[$m_to_n_func_index];

    # 損失関数の微小変化 / この層のバイアスの微小変化(バックプロパゲーションで求める)
    # 次の層の重みの傾きの転置行列とバイアスの傾きの転置行列をかけて、それぞれの要素に、活性化関数の導関数をかける
    my $forword_weights_mat = $m_to_n_func_infos->[$m_to_n_func_index + 1]{weights_mat};
    my $forword_weights_mat_transpose = SPVM::MyAIUtil->mat_transpose($forword_weights_mat);
    my $forword_biase_grads = $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index + 1];
    my $forword_biase_grads_mat = SPVM::MyAIUtil->mat_new($forword_biase_grads, $forword_biase_grads->length, 1);
    my $mul_forword_weights_transpose_mat_forword_biase_grads_mat = SPVM::MyAIUtil->mat_mul($forword_weights_mat_transpose, $forword_biase_grads_mat);
    my $mul_forword_weights_transpose_mat_forword_biase_grads_mat_values = $mul_forword_weights_transpose_mat_forword_biase_grads_mat->values;
    my $grads_outputs_to_array_sigmoid = SPVM::MyAIUtil->array_sigmoid_derivative($outputs);
    my $biase_grads = SPVM::MyAIUtil->array_mul($mul_forword_weights_transpose_mat_forword_biase_grads_mat_values, $grads_outputs_to_array_sigmoid);

    $biase_grads_in_m_to_n_funcs->[$m_to_n_func_index] = $biase_grads;
    
    # 損失関数の微小変化 / この層の重みの微小変化(バックプロパゲーションで求める)
    my $biase_grads_mat = SPVM::MyAIUtil->mat_new($biase_grads, $biase_grads->length, 1);
    my $inputs = $inputs_in_m_to_n_funcs->[$m_to_n_func_index];
    my $inputs_mat_transpose = SPVM::MyAIUtil->mat_new($inputs, 1, $inputs->length);
    my $weights_grads_mat = SPVM::MyAIUtil->mat_mul($biase_grads_mat, $inputs_mat_transpose);
    
    $weight_grads_mat_in_m_to_n_funcs->[$m_to_n_func_index] = $weights_grads_mat;
  }

  my $m_to_n_func_grad_infos = {};
  $m_to_n_func_grad_infos->{biases} = $biase_grads_in_m_to_n_funcs;
  $m_to_n_func_grad_infos->{weights_mat} = $weight_grads_mat_in_m_to_n_funcs;
  
  return $m_to_n_func_grad_infos;
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

sub dump_array {
  my ($nums) = @_;
  
  my $dump_str = join(' ', @{$nums->to_elems});
  
  print STDERR "$dump_str\n";
}

sub dump_mat {
  my ($mat) = @_;
  
  my $values_str = join(' ', @{$mat->values->to_elems});
  my $dump_str = sprintf("rows_legnth: %d, columns_length: %d, values : %s", $mat->rows_length, $mat->columns_length, $values_str);
  
  print STDERR "$dump_str\n";
}
