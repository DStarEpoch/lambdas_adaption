#ifndef LAMBDA_INFO_CONTEXT_H
#define LAMBDA_INFO_CONTEXT_H

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
#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif

typedef struct _LambdaInfoContext
{
    long start_lambda_idx;
    long end_lambda_idx;
    double ratio;

    int is_insert;
    long org_idx;
    double f_k;

    char* (*getStr)(struct _LambdaInfoContext *self);
    double (*getRank)(struct _LambdaInfoContext *self);
}LambdaInfoContext;

typedef struct _LambdaInfoContextObject{
    PyObject_HEAD
    LambdaInfoContext *context;
} LambdaInfoContextObject;

void initLambdaInfoContext(LambdaInfoContext *self, long start_lambda_idx, long end_lambda_idx, double ratio, int is_insert, long org_idx, double f_k);
void freeLambdaInfoContext(LambdaInfoContext *self);
char* getStr(LambdaInfoContext *self);

double getRank(LambdaInfoContext *self);

LambdaInfoContext* newLambdaInfoContext(long start_lambda_idx, long end_lambda_idx, double ratio, int is_insert, long org_idx, double f_k);

extern PyTypeObject LambdaInfoContextObjectType;
extern PyObject* LambdaInfoContext_createFromProperties(PyTypeObject *cls, PyObject *args);

#endif