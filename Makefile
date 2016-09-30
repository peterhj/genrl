.PHONY: all examples

all:
	cargo build --release

examples:
	cargo build --release --example train-cartpole
