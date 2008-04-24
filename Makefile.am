# $Id$

docdir = $(prefix)/share/doc/@PACKAGE@
doc_DATA = README INSTALL COPYING AUTHORS ChangeLog

liblbfgsincludedir = $(includedir)
liblbfgsinclude_HEADERS = \
	include/lbfgs.h

EXTRA_DIST = \
	autogen.sh

lib_LTLIBRARIES = liblbfgs.la
noinst_PROGRAMS = lbfgssample

liblbfgs_la_SOURCES = \
	lib/arithmetic_ansi.h \
	lib/arithmetic_sse_double.h \
	lib/arithmetic_sse_float.h \
	lib/lbfgs.c

liblbfgs_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

lbfgssample_SOURCES = \
	sample/sample.c

lbfgssample_LDADD = ./liblbfgs.la

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
