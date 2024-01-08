#ifndef SAMPLE_GENERATOR_H
#define SAMPLE_GENERATOR_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)
    #error Cython requires Python 2.6+ or Python 3.3+.
#endif

#include "structmember.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif
#include "math.h"
#include "lambda_info_context.h"

typedef struct _SampleGeneratorObject {
    PyObject_HEAD
    long lambda_num;
    long samples_per_lambda;
    double *org_u_nks;
    double *f_k;
}SampleGeneratorObject;

#endif