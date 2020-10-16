use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/network_lib";
use List::Util 'shuffle';

use SPVM 'SPVM::MyAIUtil';
use SPVM 'SPVM::Hash';
use SPVM 'SPVM::List';

# 学習率
my $learning_rate = 3;

# エポック数 - 訓練セットの実行回数
my $epoch_count = 30;

# ミニバッチサイズ
my $mini_batch_size = 10;

# 各層のニューロンの数
# 28 * 28 = 728のモノクロ画像を (入力)
# 30個の中間出力を通って        (中間出力1)
# 0～9の10個に分類する          (出力)
my $neurons_count_in_layers = SPVM::IntList->new([784, 30, 10]);

# 各層のm個の入力をn個の出力に変換する関数の情報。入力数、出力数、バイアス、重み
my $m_to_n_func_infos = SPVM::List->new([]);

# 各層のニューロン数からmからnへの変換関数の情報を作成
for (my $i = 0; $i < $neurons_count_in_layers->length - 1; $i++) {
  my $inputs_length = $neurons_count_in_layers->get($i);
  my $outputs_length = $neurons_count_in_layers->get($i + 1);
  
  # バイアスを0で初期化
  my $biases = SPVM::MyAIUtil->array_new_zero($outputs_length);

  # Xivierの初期値で重みを初期化。重みは列優先行列
  my $weights_mat = SPVM::MyAIUtil->mat_new_zero($outputs_length, $inputs_length);
  my $weights_length = $weights_mat->rows_length * $weights_mat->columns_length;
  $weights_mat->set_values(SPVM::MyAIUtil->array_create_he_init_value($weights_length, $inputs_length));
  
  # 変換関数の情報を設定
  my $m_to_n_func_info = SPVM::Hash->new([
    inputs_length => SPVM::Int->new($inputs_length),
    outputs_length => SPVM::Int->new($outputs_length),
    biases => $biases,
    weights_mat => $weights_mat,
  ]);
  $m_to_n_func_infos->push($m_to_n_func_info);
}

# MNIEST画像情報を読み込む - 入力用につかう手書きの訓練データ
my $mnist_train_image_file = "$FindBin::Bin/data/train-images-idx3-ubyte";
my $mnist_train_image_info = load_mnist_train_image_file($mnist_train_image_file);

# MNIESTラベル情報を読み込む - 手書きの訓練データの期待される出力
my $mnist_train_label_file = "$FindBin::Bin/data/train-labels-idx1-ubyte";
my $mnist_train_label_info = load_mnist_train_label_file($mnist_train_label_file);

# MNIEST画像情報をSPVMデータに変換
my $mnist_train_image_info_spvm = SPVM::Hash->new([
  items_count => SPVM::Int->new($mnist_train_image_info->{items_count}),
  rows_count => SPVM::Int->new($mnist_train_image_info->{rows_count}),
  columns_count => SPVM::Int->new($mnist_train_image_info->{columns_count}),
  data => SPVM::new_byte_array_from_bin($mnist_train_image_info->{data}),
]);

# MNIESTラベル情報をSPVMデータに変換
my $mnist_train_label_info_spvm = SPVM::Hash->new([
  items_count => SPVM::Int->new($mnist_train_label_info->{items_count}),
  label_numbers => SPVM::IntList->new($mnist_train_label_info->{label_numbers}),
]);

# 訓練データのインデックス(最初の4万枚だけを訓練用データとして利用する。残りの1万枚は検証用データとする)
my @training_data_indexes = (0 .. 40000);

# ミニバッチ単位における各変換関数の情報
my $m_to_n_func_mini_batch_infos = SPVM::List->new_len($m_to_n_func_infos->length);

# ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
# メモリ領域を繰り返しつかうためここで初期化
for (my $m_to_n_func_index = 0; $m_to_n_func_index < $m_to_n_func_infos->length; $m_to_n_func_index++) {
  my $m_to_n_func_info = $m_to_n_func_infos->get($m_to_n_func_index);
  my $biases = $m_to_n_func_info->get('biases');
  my $weights_mat = $m_to_n_func_info->get('weights_mat');
  
  # バイアスの長さ
  my $biases_length = $biases->length;

  # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
  my $biase_grad_totals = SPVM::MyAIUtil->array_new_zero($biases_length);

  # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
  my $weight_grad_totals_mat = SPVM::MyAIUtil->mat_new_zero($weights_mat->rows_length, $weights_mat->columns_length);
  
  my $hash = SPVM::Hash->new([
    'biase_grad_totals' => $biase_grad_totals,
    'weight_grad_totals_mat' => $weight_grad_totals_mat,
  ]);
  
  $m_to_n_func_mini_batch_infos->set($m_to_n_func_index => $hash);
}

# エポックの回数だけ訓練セットを実行
for (my $epoch_index = 0; $epoch_index < $epoch_count; $epoch_index++) {
  
  # 訓練データのインデックスをシャッフル(ランダムに学習させた方が汎用化するらしい)
  my @training_data_indexes_shuffle = shuffle @training_data_indexes;
  
  my $count = 0;
  
  # ミニバッチサイズ単位で学習
  my $backprop_count = 0;
  
  while (my @indexed_for_mini_batch = splice(@training_data_indexes_shuffle, 0, $mini_batch_size)) {
    
    # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きの合計を0で初期化
    for (my $m_to_n_func_index = 0; $m_to_n_func_index < $m_to_n_func_mini_batch_infos->length; $m_to_n_func_index++) {
      my $m_to_n_func_info = $m_to_n_func_infos->get($m_to_n_func_index);
      my $biases = $m_to_n_func_info->get('biases');
      my $weights_mat = $m_to_n_func_info->get('weights_mat');
      
      # ミニバッチにおける各変換関数のバイアスの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->set(biase_grad_totals => SPVM::MyAIUtil->array_new_zero($m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('biase_grad_totals')->length));
      
      # ミニバッチにおける各変換関数の重みの傾きの合計を0で初期化して作成
      $m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get("weight_grad_totals_mat")->set_values(SPVM::MyAIUtil->array_new_zero($m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('weight_grad_totals_mat')->values->length));
    }
    
    for my $training_data_index (@indexed_for_mini_batch) {
      # バックプロパゲーションを使って重みとバイアスの損失関数に関する傾きを取得
      my $m_to_n_func_grad_infos = SPVM::MyAIUtil->backprop($m_to_n_func_infos, $mnist_train_image_info_spvm, $mnist_train_label_info_spvm, $training_data_index);
      
      # バイアスの損失関数に関する傾き
      my $biase_grads = $m_to_n_func_grad_infos->get('biases');
      
      # 重みの損失関数に関する傾き
      my $weight_grads_mat = $m_to_n_func_grad_infos->get('weights_mat');

      # ミニバッチにおける各変換関数のバイアスの傾きの合計とミニバッチにおける各変換関数の重みの傾きを加算
      for (my $m_to_n_func_index = 0; $m_to_n_func_index < $m_to_n_func_mini_batch_infos->length; $m_to_n_func_index++) {
        my $m_to_n_func_info = $m_to_n_func_infos->get($m_to_n_func_index);
        
        # ミニバッチにおける各変換関数のバイアスの傾きを加算
        SPVM::MyAIUtil->array_add_inplace($m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('biase_grad_totals'), $biase_grads->get($m_to_n_func_index));

        # ミニバッチにおける各変換関数の重みの傾きを加算
        SPVM::MyAIUtil->array_add_inplace($m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('weight_grad_totals_mat')->values, $weight_grads_mat->get($m_to_n_func_index)->values);
      }
    }

    # 各変換関数のバイアスと重みをミニバッチの傾きの合計を使って更新
    for (my $m_to_n_func_index = 0; $m_to_n_func_index < $m_to_n_func_infos->length; $m_to_n_func_index++) {
      
      # 各変換関数のバイアスを更新(学習率を考慮し、ミニバッチ数で割る)
      SPVM::MyAIUtil->update_params($m_to_n_func_infos->get($m_to_n_func_index)->get('biases'), $m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('biase_grad_totals'), $learning_rate, $mini_batch_size);
      
      # 各変換関数の重みを更新(学習率を考慮し、傾きの合計をミニバッチ数で、ミニバッチ数で割る)
      SPVM::MyAIUtil->update_params($m_to_n_func_infos->get($m_to_n_func_index)->get('weights_mat')->values, $m_to_n_func_mini_batch_infos->get($m_to_n_func_index)->get('weight_grad_totals_mat')->values, $learning_rate, $mini_batch_size);
    }
  }
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
