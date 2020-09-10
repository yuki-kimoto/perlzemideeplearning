use strict;
use warnings;

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
  my $mean_square_error = 0.5 * $total_pow2;
  
  return $mean_square_error;
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
