CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PWD != dirname $(realpath $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

PYTHON_VER := cpython-3.13.1+20250115-x86_64_v3-unknown-linux-gnu-install_only_stripped.tar.gz
PYTHON_DIR := $(PWD)/.python

.PHONY: python
python: $(PYTHON_DIR)/.python-packages $(PYTHON_DIR)/bin/fpp

$(PYTHON_DIR)/bin/fpp: fpp/fpp
	mkdir -p $(PYTHON_DIR)/bin
	cp fpp/fpp $(PYTHON_DIR)/bin
	touch $@

$(PYTHON_DIR)/.python: python/$(PYTHON_VER)
	mkdir -p $(PYTHON_DIR)
	tar xf $^ --strip-components=1 --directory $(PYTHON_DIR)
	touch $@

$(PYTHON_DIR)/.python-packages: $(PYTHON_DIR)/.python
	$(PYTHON_DIR)/bin/python3 -m ensurepip
	$(PYTHON_DIR)/bin/pip3 install --no-index --find-link=python/wheels -r requirements.txt
	touch $@

.PHONY: python-packages-download
python-packages-download:
	$(PYTHON_DIR)/bin/pip3 download --destination-directory=python/wheels -r requirements.txt

clean:
	$(RM) -rf $(PYTHON_DIR)
