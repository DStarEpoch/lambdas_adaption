#ifndef DP_INFO_H
#define DP_INFO_H

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


typedef struct _DPInfo
{
    struct _DPInfo *parent;
    long latest_insert_idx;
    double cost;

    long (*getSequenceLength)(struct _DPInfo *self);
    long* (*getSequence)(struct _DPInfo *self);
    void (*setParent)(struct _DPInfo *self, struct _DPInfo *parent);
}DPInfo;

typedef struct _DPInfoObject{
    PyObject_HEAD
    DPInfo *dp_info;
} DPInfoObject;

long getDPInfoSequenceLength(DPInfo *self);
long *getDPInfoSequence(DPInfo *self);
void setDPInfoParent(DPInfo *self, DPInfo *parent);

void initDPInfo(DPInfo *self, long latest_insert_idx);
void freeDPInfo(DPInfo *self);

DPInfo* newDPInfo(long latest_insert_idx);

extern PyTypeObject DPInfoType;

#endif