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
