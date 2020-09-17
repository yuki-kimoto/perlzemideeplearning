use strict;
use warnings;

# ベクトルの差
sub vec_sub {
  my ($vec1, $vec2) = @_;

  my $vec_sub = [];

  for (my $i = 0; $i < @$vec1; $i++) {
    $vec_sub->[$i] = $vec1->[$i] - $vec2->[$i];
  }
  
  return $vec_sub;
}

my $vec1 = [1, 2, 3];
my $vec2 = [4, 5, 6];

# ベクトルの差
my $vec_sub = vec_sub($vec1, $vec2);

# 5 7 9
print "@$vec_sub\n";


__END__
use strict;
use warnings;

# ベクトルの和
sub vec_add {
  my ($vec1, $vec2) = @_;

  my $vec_add = [];

  for (my $i = 0; $i < @$vec1; $i++) {
    $vec_add->[$i] = $vec1->[$i] + $vec2->[$i];
  }
  
  return $vec_add;
}

my $vec1 = [1, 2, 3];
my $vec2 = [4, 5, 6];

# ベクトルの和
my $vec_add = vec_add($vec1, $vec2);

# 5 7 9
print "@$vec_add\n";

__END__

use strict;
use warnings;

sub mat_add {
  my ($mat1, $mat2) = @_;
  
  my $mat_out = {};
  $mat_out->{rows_length} = $mat1->{rows_length};
  $mat_out->{columns_length} = $mat2->{columns_length};
  
  for (my $i = 0; $i < @{$mat1->{values}}; $i++) {
    $mat_out->{values}->[$i] = $mat1->{values}->[$i] + $mat2->{values}->[$i];
  }
  
  return $mat_out;
}

sub mat_sub {
  my ($mat1, $mat2) = @_;
  
  my $mat_out = {};
  $mat_out->{rows_length} = $mat1->{rows_length};
  $mat_out->{columns_length} = $mat2->{columns_length};
  
  for (my $i = 0; $i < @{$mat1->{values}}; $i++) {
    $mat_out->{values}->[$i] = $mat1->{values}->[$i] - $mat2->{values}->[$i];
  }
  
  return $mat_out;
}

my $mat1 = {
  values => [1, 2, 3, 4, 5, 6],
  rows_length => 3,
  columns_length => 2,
};

my $mat2 = {
  values => [7, 8, 9, 10, 11, 12],
  rows_length => 3,
  columns_length => 2,
};

my $mat_add = mat_add($mat1, $mat2);

# [Matrix Add]Row:3, Column:2, Values:8 10 12 14 16 18
print "[Matrix Add]Row:$mat_add->{rows_length}, Column:$mat_add->{columns_length}, Values:@{$mat_add->{values}}\n";


my $mat_sub = mat_sub($mat1, $mat2);

# [Matrix Subtract]Row:3, Column:2, Values:-6 -6 -6 -6 -6 -6
print "[Matrix Subtract]Row:$mat_sub->{rows_length}, Column:$mat_sub->{columns_length}, Values:@{$mat_sub->{values}}\n";

__END__

use strict;
use warnings;

sub transpose {
  my ($mat) = @_;
  
  my $rows_length = $mat->{rows_length};
  my $columns_length = $mat->{columns_length};
  my $length = $rows_length * $columns_length;
  
  my $mat_trans ={};
  $mat_trans->{rows_length} = $columns_length;
  $mat_trans->{columns_length} = $rows_length;
  
  my $values = $mat->{values};
  my $mat_trans_values = [];
  
  for (my $row_index = 0; $row_index < $rows_length; $row_index++) {
    for (my $column_index = 0; $column_index < $columns_length; $column_index++) {
      $mat_trans_values->[$row_index * $columns_length + $column_index] = $values->[$column_index * $rows_length+ $row_index];
    }
  }
  $mat_trans->{values} = $mat_trans_values;
  
  return $mat_trans;
}

my $mat = {
  values => [1, 2, 3, 4, 5, 6],
  rows_length => 3,
  columns_length => 2,
};

my $mat_trans = transpose($mat);

print "Row:$mat_trans->{rows_length}, Column:$mat_trans->{columns_length}, Values:@{$mat_trans->{values}}\n";

__END__

use strict;
use warnings;

sub probabilize_output {
  my ($outputs_length, $desired_value) = @_;
  
  my $probabilized_outputs = [];
  for (my $i = 0; $i < $outputs_length; $i++) {
    $probabilized_outputs->[$i] = $desired_value == $i ? 1 : 0;
  }
  
  return $probabilized_outputs;
}

my $outputs_length = 10;
my $desired_value = 4;

my $probabilized_outputs = probabilize_output($outputs_length, $desired_value);

# 0 0 0 0 1 0 0 0 0 0
print "@$probabilized_outputs\n";

__END__

use strict;
use warnings;

# 重み(3行2列の行列)
# 1 4
# 2 5
# 3 6
my $weights = [1, 2, 3, 4, 5, 6];
my $weights_rows_length = 3;
my $weights_columns_length = 2;

# 入力ベクトル(2行1列の行列)
# 7
# 8
my $inputs = [7, 8];
my $inputs_rows_length = 2;
my $inputs_columns_length = 1;

# 計算方法
# 1 * 7 + 4 * 8
# 2 * 7 + 5 * 8
# 3 * 7 + 6 * 8
my $outputs = [];

# 行列の積の計算
for(my $row = 0; $row < $weights_rows_length; $row++) {
  for(my $col = 0; $col < $inputs_columns_length; $col++) {
    for(my $incol = 0; $incol < $weights_columns_length; $incol++) {
      $outputs->[$row + $col * $inputs_rows_length]
       += $weights->[$row + $incol * $weights_rows_length] * $inputs->[$incol + $col * $inputs_rows_length];
    }
  }
}

# 39 54 69
print "@$outputs\n";

__END__

use strict;
use warnings;

my $first_inputs = [0.1, 0.2];

# 隠れ層の重みとバイアス
my $layers = [
  # 隠れ層0層(2入力を3出力へ)
  {
    weights => [
      0.6, 0.2, 0.4,
      0.4, 0.3, 0.7
    ],
    weights_rows_length => 3,
    weights_columns_length => 2,
    biases => [0.5, 0.2, 0.8]
  },
  # 隠れ層1層(3入力を2出力へ)
  {
    weights => [
      0.8, 0.2, 0.2,
      0.2, 0.1, 0.6
    ],
    weights_rows_length => 2,
    weights_columns_length => 3,
    biases => [0.5, 0.1]
  },
];

# 初期入力から最終出力を得る
my $inputs = $first_inputs;
my $outputs;
for (my $i = 0; $i < @$layers; $i++) {
  my $layer = $layers->[$i];
  
  # 重み
  my $weights = $layer->{weights};
  my $weights_rows_length = $layer->{weights_rows_length};
  my $weights_columns_length = $layer->{weights_columns_length};
  
  # バイアス
  my $biases = $layer->{biases};
  
  # 出力 = 重み行列 * 入力 + バイアス
  my $mul_weight_inputs = mul_mat($weights, $weights_rows_length, $weights_columns_length, $inputs);
  $outputs = add_vec($mul_weight_inputs, $biases);
  
  # 活性化関数を適用
  my $activate_outputs = [];
  for (my $i = 0; $i < @$outputs; $i++) {
    $activate_outputs->[$i] = relu($outputs->[$i]);
  }
  
  # 出力を次の入力へ
  $inputs = $activate_outputs;
}

# 1.166 0.872
print "@$outputs\n";

# 活性化関数 ReLU
sub relu {
  my ($x) = @_;
  
  my $relu = $x * ($x > 0.0);
  
  return $relu;
}

# ベクトルの和
sub add_vec {
  my ($mul_weight_inputs, $biases) = @_;
  
  my $outputs = [];
  for (my $i = 0; $i < @$mul_weight_inputs; $i++) {
    $outputs->[$i] = $mul_weight_inputs->[$i] + $biases->[$i];
  }
  
  return $outputs;
}

# 行列の積(行列とベクトルの掛け算)
sub mul_mat {
  my ($weights, $weights_rows_length, $weights_columns_length, $inputs) = @_;
  
  my $inputs_rows_length = @$inputs;
  my $inputs_columns_length = 1;

  my $outputs = [];
  
  # 行列の積の計算
  for(my $row = 0; $row < $weights_rows_length; $row++) {
    for(my $col = 0; $col < $inputs_columns_length; $col++) {
      for(my $incol = 0; $incol < $weights_columns_length; $incol++) {
        $outputs->[$row + $col * $inputs_rows_length]
         += $weights->[$row + $incol * $weights_rows_length] * $inputs->[$incol + $col * $inputs_rows_length];
      }
    }
  }
  
  return $outputs;
}

__END__

# 重み(3行2列の行列)
# 1 4
# 2 5
# 3 6
my $weights = [1, 2, 3, 4, 5, 6];
my $weights_rows_length = 3;
my $weights_columns_length = 2;

# 入力ベクトル(2行1列の行列)
# 7
# 8
my $inputs = [7, 8];
my $inputs_rows_length = 2;
my $inputs_columns_length = 1;

# 計算方法
# 1 * 7 + 4 * 8
# 2 * 7 + 5 * 8
# 3 * 7 + 6 * 8
my $outputs = [];

# 行列の積の計算
for(my $row = 0; $row < $weights_rows_length; $row++) {
  for(my $col = 0; $col < $inputs_columns_length; $col++) {
    for(my $incol = 0; $incol < $weights_columns_length; $incol++) {
      $outputs->[$row + $col * $inputs_rows_length]
       += $weights->[$row + $incol * $weights_rows_length] * $inputs->[$incol + $col * $inputs_rows_length];
    }
  }
}

# 40 54 69
print "@$outputs\n";

__END__

use strict;
use warnings;

# 重み行列と入力ベクトルの行列積の結果のベクトル
my $mul_weight_inputs = [4, 5, 6, 7];

# バイアスのベクトル
my $biases = [3, 6, 9, 2];

# 出力ベクトル
my $outputs = [];

for (my $i = 0; $i < @$mul_weight_inputs; $i++) {
  $outputs->[$i] = $mul_weight_inputs->[$i] + $biases->[$i];
}

print "@$outputs\n";

__END__

use strict;
use warnings;

# 重み行列と入力ベクトルの行列積の結果のベクトル
my $mul_weight_inputs = [4, 5, 6, 7];

# バイアスのベクトル
my $biases = [3, 6, 9, 2];

# 出力ベクトル
my $outputs = [];

for (my $i = 0; $i < @$vec1; $i++) {
  $outputs->[$i] = $mul_weight_inputs->[$i] + $biases->[$i];
}

print "@$outputs\n";

__END__

sub relu_derivative {
  my ($x) = @_;
  
  my $relu_derivative = 1 * ($x > 0.0);
  
  return $relu_derivative;
}

my $value1 = 0.7;
my $relu_derivative1 = relu_derivative($value1);

print "$relu_derivative1\n";

my $value2 = -0.4;
my $relu_derivative2 = relu_derivative($value2);

print "$relu_derivative2\n";

__END__

use strict;
use warnings;

# ReLU関数
sub relu {
  my ($x) = @_;
  
  my $relu = $x * ($x > 0.0);
  
  return $relu;
}

my $value1 = 0.7;
my $relu1 = relu($value1);

print "$relu1\n";

my $value2 = -0.4;
my $relu2 = relu($value2);

print "$relu2\n";

__END__

sub cross_entropy_cost_delta {
  my ($outputs, $activate_outputs, $desired_outputs) = @_;

  if (@$activate_outputs != @$desired_outputs) {
    die "Outputs length is different from Desired length";
  }
  
  my $cross_entropy_cost_delta = [];
  for (my $i = 0; $i < @$activate_outputs; $i++) {
    $cross_entropy_cost_delta->[$i] = $activate_outputs->[$i] - $desired_outputs->[$i];
  }
  
  return $cross_entropy_cost_delta;
}

my $activate_outputs = [0.6, 0, 0.2];
my $desired_outputs = [1, 0, 0];
my $cross_entropy_cost = cross_entropy_cost_delta(undef, $activate_outputs, $desired_outputs);

print "@$cross_entropy_cost\n";

__END__

# 交差エントロピー誤差
sub cross_entropy_cost {
  my ($outputs, $desired_outputs) = @_;
  
  if (@$outputs != @$desired_outputs) {
    die "Outputs length is different from Desired length";
  }
  
  my $cross_entropy_cost = 0;
  
  for (my $i = 0; $i < @$outputs; $i++) {
    $cross_entropy_cost += -$desired_outputs->[$i] * log($outputs->[$i]) - (1 - $desired_outputs->[$i]) * log(1 - $outputs->[$i]);
  }
  
  return $cross_entropy_cost;
}

my $outputs = [0.7, 0.2, 0.1];
my $desired_outputs = [1, 0, 0];
my $cross_entropy_cost = cross_entropy_cost($outputs, $desired_outputs);

print "$cross_entropy_cost\n";

__END__

sub cross_entropy_cost {
  my ($y, $hidden_outputs, $desired_outputs) = @_;
  
  (np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

}

__END__

# 二乗和誤差
sub sum_of_square_error {
  my ($outputs, $desired_outputs) = @_;
  
  if (@$outputs != @$desired_outputs) {
    die "Outputs length is different from Desired length";
  }
  
  my $total_pow2 = 0;
  for (my $i = 0; $i < @$outputs; $i++) {
    $total_pow2 += ($outputs->[$i] - $desired_outputs->[$i]) ** 2;
  }
  my $sum_of_square_error = 0.5 * $total_pow2;
  
  return $sum_of_square_error;
}

my $outputs = [0.7, 0.2, 0.1];
my $desired_outputs = [1, 0, 0];
my $sum_of_square_error = sum_of_square_error($outputs, $desired_outputs);

print "$sum_of_square_error\n";

__END__
my $ret = sigmoid(3);

print "$ret\n";

sub sigmoid {
  my ($z) = @_;
  
  my $sigmoid = 1.0 / (1.0 + exp(-$z));
  
  return $sigmoid;
}
