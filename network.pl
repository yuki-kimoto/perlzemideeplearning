use strict;
use warnings;

use JSON::PP;

sub vec_

# ベクトルのマイナス
sub vec_minus {
  my ($vec) = @_;
  
  my $vec_out = [];
  for (my $i = 0; $i < @$vec; $i++) {
    $vec_out->[$i] = -$vec->[$i];
  }
  
  return $vec_out;
}

# ベクトルの要素の和を求める
sub vec_sum {
  my ($vec) = @_;
  
  my $total = 0;
  for (my $i = 0; $i < @$vec; $i++) {
    $total += $vec->[$i];
  }
  
  return $total;
}

# ベクトルの長さを求める
sub vec_len {
  my ($vec) = @_;
  
  my $squared_sum = 0;
  for (my $i = 0; $i < @$vec; $i++) {
    $squared_sum += $vec->[$i] * $vec->[$i];
  }
  
  my $vec_len = sqrt($squared_sum);
  
  return $vec_len;
}

# ベクトルの足し算
sub vec_add {
  my ($vec1, $vec2) = @_;
  
  my $vec3 = [];
  for (my $i = 0; $i < @$vec1; $i++) {
    $vec3->[$i] = $vec1->[$i] + $vec1->[$i];
  }
  
  return $vec3;
}

# ベクトルの引き算
sub vec_sub {
  my ($vec1, $vec2) = @_;
  
  my $vec3 = [];
  for (my $i = 0; $i < @$vec1; $i++) {
    $vec3->[$i] = $vec1->[$i] - $vec1->[$i];
  }
  
  return $vec3;
}

# ベクトルのlog
sub vec_log {
  my ($vec) = @_;
  
  my $vec_out = [];
  for (my $i = 0; $i < @$vec; $i++) {
    $vec_out->[$i] = log($vec->[$i]);
  }
  
  return $vec_out;
}

# ベクトルのexp
sub vec_exp {
  my ($vec) = @_;
  
  my $vec_out = [];
  for (my $i = 0; $i < @$vec; $i++) {
    $vec_out->[$i] = exp($vec->[$i]);
  }
  
  return $vec_out;
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
