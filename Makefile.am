# $Id$

SUBDIRS = lib sample

docdir = $(prefix)/share/doc/@PACKAGE@
doc_DATA = README INSTALL COPYING AUTHORS ChangeLog NEWS

EXTRA_DIST = \
	autogen.sh \
	lbfgs.sln

##############################
# googletest support
##############################
all-local: googletest/.built
check-recursive: googletest/.built

googletest/.built: googletest/Makefile
	cmake --build $(@D)
	touch $@
googletest/Makefile: googletest/CMakeLists.txt
	cd $(@D); cmake .
