#!/usr/bin/perl

use strict;
use warnings;
use File::Basename;

if (@ARGV < 1) {
    die "usage: pre-prepare.pl <input>\n";
}

my $input = $ARGV[0];
my $basename = basename($input);
my $dirname = dirname($input);
my $output = $dirname . '/_' . $basename;

open(my $rinput, "<", $input) or die "unable to open input: $!";
open(my $routput, ">", $output) or die "unable to open output: $!";

while (my $line = <$rinput>) {
    chomp $line;
    my @cols = split /,/, $line;

    if (@cols >= 3) {
        my $col = join(",", @cols[3..$#cols]);
        $col =~ s/"//g;
        $cols[3] = "\"" . $col . "\"";
        @cols = @cols[0..3];
    }
    
    my $mod_line = join(',', @cols);
    print $routput $mod_line . "\n";
}

close $rinput;
close $routput;

print "the file $output was created with the modifications.\n";
