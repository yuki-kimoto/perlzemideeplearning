use strict;
use warnings;
use List::Util 'shuffle';

# インデックスのシャッフル
my @indexes = (0 .. 9);
my @shuffled_indexes = shuffle @indexes;

# 画像情報を想定したデータをランダムな順で取得
my @training_datas = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k');
for my $index (@shuffled_indexes) {
  my $training_data = $training_datas[$index];
  print "$training_data\n";
}
