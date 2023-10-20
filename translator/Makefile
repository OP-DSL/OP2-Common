include ../makefiles/common.mk

CC=gcc
CXX=g++

undefine CFLAGS
undefine CXXFLAGS

PYTHON_VER := 3.12.0

.PHONY: python
python: .python-packages .fpp

.fpp: fpp/fpp
	mkdir -p $(OP2_BUILD_DIR)/bin
	cp fpp/fpp $(OP2_BUILD_DIR)/bin
	touch .fpp

.python: python/Python-$(PYTHON_VER).tar.xz
	tar xf $^ --directory python
	cd python/Python-$(PYTHON_VER); ./configure --prefix=$(OP2_BUILD_DIR)
	cd python/Python-$(PYTHON_VER); $(MAKE) install
	touch .python

.python-packages: .python
	$(OP2_BUILD_DIR)/bin/pip3 install --upgrade --no-index --find-link=python/wheels pip
	$(OP2_BUILD_DIR)/bin/pip3 install --no-index --find-link=python/wheels -r requirements.txt
	touch .python-packages

.PHONY: python-packages-download
python-packages-download:
	$(OP2_BUILD_DIR)/bin/pip3 download --destination-directory=python/wheels -r requirements.txt

clean:
	$(RM) -rf .python .python-packages .fpp
	$(RM) -rf python/Python-$(PYTHON_VER)
