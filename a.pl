use strict;
use warnings;

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
