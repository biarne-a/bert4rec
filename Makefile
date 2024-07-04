# -*- mode: makefile -*-
install_mac_m2:
	conda create --name=bert4rec-m2 python=3.9
	conda activate bert4rec-m2
	conda install -c apple tensorflow-deps
	pip install tensorflow-macos==2.14 tensorflow-metal
	conda install -c conda-forge --file requirements.in


install:
	conda create --name=bert4rec python=3.9
	conda activate bert4rec
	conda install -c conda-forge tensorflow
	conda install -c conda-forge --file requirements.in


clean:
	./scripts/clean.sh
