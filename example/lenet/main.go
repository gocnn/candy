package main

import (
	"flag"
	"fmt"
	"os"
)

func die(v ...any) {
	fmt.Fprintln(os.Stderr, v...)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		die("Usage: lenet <train|infer> [options]")
	}
	sub := os.Args[1]
	switch sub {
	case "train":
		fs := flag.NewFlagSet("train", flag.ExitOnError)
		dir := fs.String("data", "./data", "MNIST data directory")
		out := fs.String("out", "lenet.npz", "output weights file")
		_ = fs.Parse(os.Args[2:])
		if err := RunTrain(*dir, *out); err != nil {
			die(err)
		}
	case "infer":
		fs := flag.NewFlagSet("infer", flag.ExitOnError)
		w := fs.String("weights", "lenet.npz", "path to weights npz")
		f := fs.String("file", "", "path to one .pgm file")
		dir := fs.String("path", ".", "directory containing .pgm files")
		inv := fs.Bool("invert", false, "invert grayscale (1-p)")
		auto := fs.Bool("auto-invert", true, "auto invert if image looks white-on-black")
		_ = fs.Parse(os.Args[2:])
		if err := RunInfer(*w, *f, *dir, *inv, *auto); err != nil {
			die(err)
		}
	default:
		die("Unknown subcommand:", sub, "\nUsage: lenet <train|infer> [options]")
	}
}
