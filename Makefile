# -*- mode: makefile -*-
build_preprocess:
	docker build -t northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/bert4rec-preprocess -f bert4rec/preprocess/Dockerfile .
	cat credentials.json | docker login -u _json_key --password-stdin https://northamerica-northeast1-docker.pkg.dev
	docker push northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/bert4rec-preprocess


install_mac_m2:
    CONDA_SUBDIR=osx-arm64 conda env create -n bert4rec-m2 -f conda-env.yaml


install:
	conda env create -n bert4rec -f conda-env.yaml


clean:
	./scripts/clean.sh
