# $Id$

EXTRA_DIST = \
	lib.vcxproj

liblbfgsincludedir = $(includedir)
liblbfgsinclude_HEADERS = \
	../include/lbfgs.h

lib_LTLIBRARIES = liblbfgs.la

liblbfgs_la_SOURCES = \
	arithmetic_ansi.h \
	arithmetic_sse_double.h \
	arithmetic_sse_float.h \
	../include/lbfgs.h \
	lbfgs.c

liblbfgs_la_LDFLAGS = \
	-no-undefined \
	-release @VERSION@

bin_PROGRAMS = lbfgstest
TESTS = lbfgstest

lbfgstest_SOURCES = lbfgs_unittests.cc
lbfgstest_LDFLAGS = $(GTEST_LDFLAGS)
lbfgstest_LDADD = liblbfgs.la $(GTEST_LIBS)
lbfgstest_CPPFLAGS = -I./include $(GTEST_CPPFLAGS)

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
