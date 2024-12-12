include ../makefiles/common.mk

CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PYPY_VER := 3.10-v7.3.16-linux64

.PHONY: python
python: $(OP2_BUILD_DIR)/.python-packages $(OP2_BUILD_DIR)/.fpp

$(OP2_BUILD_DIR)/.fpp: fpp/fpp
	mkdir -p $(OP2_BUILD_DIR)/bin
	cp fpp/fpp $(OP2_BUILD_DIR)/bin
	touch $(OP2_BUILD_DIR)/.fpp

$(OP2_BUILD_DIR)/.python: python/pypy$(PYPY_VER).tar.bz2
	tar xf $^ --strip-components=1 --directory $(OP2_BUILD_DIR)
	touch $(OP2_BUILD_DIR)/.python

$(OP2_BUILD_DIR)/.python-packages: $(OP2_BUILD_DIR)/.python
	$(OP2_BUILD_DIR)/bin/python3 -m ensurepip
	$(OP2_BUILD_DIR)/bin/pip3 install --no-index --find-link=python/wheels -r requirements.txt
	touch $(OP2_BUILD_DIR)/.python-packages

.PHONY: python-packages-download
python-packages-download:
	$(OP2_BUILD_DIR)/bin/pip3 download --destination-directory=python/wheels -r requirements.txt

clean:
	$(RM) -rf $(OP2_BUILD_DIR)/.python $(OP2_BUILD_DIR)/.python-packages $(OP2_BUILD_DIR)/.fpp
