/*
 *      ANSI C implementation of vector operations.
 *
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id$ */

#include <stdlib.h>
#include <memory.h>

#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <memory.h>

#include <yepCore.h>

#if     LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
#define fsigndiff(x, y) (((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
#else
#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
#endif/*LBFGS_IEEE_FLOAT*/

inline static void* vecalloc(size_t size)
{
#if     defined(_MSC_VER)
    void *memblock = _aligned_malloc(size, 16);
#elif   defined(__APPLE__)  /* OS X always aligns on 16-byte boundaries */
    void *memblock = malloc(size);
#else
    void *memblock = NULL, *p = NULL;
    if (posix_memalign(&p, 16, size) == 0) {
        memblock = p;
    }
#endif
    if (memblock != NULL) {
        memset(memblock, 0, size);
    }
    return memblock;
}

inline static void vecfree(void *memblock)
{
#ifdef	_MSC_VER
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

inline static void vecset(lbfgsfloatval_t *x, const lbfgsfloatval_t c, const int n)
{
    int i;
    
    for (i = 0;i < n;++i) {
        x[i] = c;
    }
}

inline static void veccpy(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{
    int i;

    for (i = 0;i < n;++i) {
        y[i] = x[i];
    }
}

inline static void vecncpy(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_Negate_V64f_V64f(x, y, n);
#else
    yepCore_Negate_V32f_V32f(x, y, n);
#endif
}

inline static void vecadd(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const lbfgsfloatval_t c, const int n)
{
    int i;

    for (i = 0;i < n;++i) {
        y[i] += c * x[i];
    }
}

inline static void vecdiff(lbfgsfloatval_t *z, const lbfgsfloatval_t *x, const lbfgsfloatval_t *y, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_Subtract_V64fV64f_V64f(x, y, z, n);
#else
    yepCore_Subtract_V32fV32f_V32f(x, y, z, n);
#endif
}

inline static void vecscale(lbfgsfloatval_t *y, const lbfgsfloatval_t c, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_Multiply_IV64fS64f_IV64f(y, c, n);
#else
    yepCore_Mutiply_IV32fS32f_IV32f(y, c, n);
#endif
}

inline static void vecmul(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_Multiply_IV64fV64f_IV64f(y, x, n);
#else
    yepCore_Mutiply_IV32fV32f_IV32f(y, x, n);
#endif
}

inline static void vecdot(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const lbfgsfloatval_t *y, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_DotProduct_V64fV64f_S64f(x, y, s, n);
#else
    yepCore_DotProduct_V32fV32f_S32f(x, y, s, n);
#endif
}

inline static void vec2norm(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const int n)
{
#if LBFGS_FLOAT == 64
    yepCore_SumSquares_V64f_S64f(x, s, n);
#else
    yepCore_SumSquares_V32f_S32f(x, s, n);
#endif
    *s = (lbfgsfloatval_t)sqrt(*s);
}

inline static void vec2norminv(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const int n)
{
    vec2norm(s, x, n);
    *s = (lbfgsfloatval_t)(1.0 / *s);
}
