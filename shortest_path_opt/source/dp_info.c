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

#include "dp_info.h"

int getDPInfoSequenceLength(DPInfo *self) {
    if (self == NULL)
        return 0;
    int length = getDPInfoSequenceLength(self->parent) + 1;
    return length;
}

int *getDPInfoSequence(DPInfo *self) {
    if (self == NULL)
        return NULL;
    int seq_length = getDPInfoSequenceLength(self);
    int *ret_list = (int *)malloc(sizeof(int) * seq_length);
    if (self->parent == NULL) {
        ret_list[0] = self->latest_insert_idx;
        return ret_list;
    }
    int *parent_seq = getDPInfoSequence(self->parent);
    for (int i = 0; i < seq_length - 1; i++) {
        ret_list[i] = parent_seq[i];
    }
    ret_list[seq_length - 1] = self->latest_insert_idx;
    return ret_list;
}

void setDPInfoParent(DPInfo *self, DPInfo *parent) {
    self->parent = parent;
}

void initDPInfo(DPInfo *self, int latest_insert_idx) {
    self->parent = NULL;
    self->latest_insert_idx = latest_insert_idx;
    // init cost with infinity
    self->cost = INFINITY;

    self->getSequenceLength = getDPInfoSequenceLength;
    self->getSequence = getDPInfoSequence;
    self->setParent = setDPInfoParent;
}

void freeDPInfo(DPInfo *self) {
    if (self == NULL)
        return;
    freeDPInfo(self->parent);
    free(self);
}

DPInfo *newDPInfo(int latest_insert_idx) {
    DPInfo *ret = (DPInfo *)malloc(sizeof(DPInfo));
    initDPInfo(ret, latest_insert_idx);
    return ret;
}

// 封装DPInfo为Python对象
typedef struct _DPInfoObject{
    PyObject_HEAD
    DPInfo *dp_info;
} DPInfoObject;

static void
DPInfo_dealloc(DPInfoObject *self)
{
    freeDPInfo(self->dp_info);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
DPInfo_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DPInfoObject *self;
    self = (DPInfoObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->dp_info = NULL;
    }
    return (PyObject *) self;
}

static int
DPInfo_init(DPInfoObject *self, PyObject *args, PyObject *kwds)
{
    int latest_insert_idx;
    if (!PyArg_ParseTuple(args, "i", &latest_insert_idx))
        return -1;
    self->dp_info = newDPInfo(latest_insert_idx);
    return 0;
}

static PyObject *
DPInfo_get_cost(DPInfoObject *self, void *closure)
{
    return Py_BuildValue("f", self->dp_info->cost);
}

static PyObject *
DPInfo_set_cost(DPInfoObject *self, PyObject *value, void *closure)
{
    double cost;
    if (!PyArg_ParseTuple(value, "d", &cost))
        return NULL;
    self->dp_info->cost = cost;
    return 0;
}

static PyObject *
DPInfo_get_latest_insert_idx(DPInfoObject *self, void *closure)
{
    return Py_BuildValue("i", self->dp_info->latest_insert_idx);
}

static PyObject *
DPInfo_get_sequence(DPInfoObject *self, void *closure)
{
    int *seq = self->dp_info->getSequence(self->dp_info);
    int seq_length = self->dp_info->getSequenceLength(self->dp_info);
    PyObject *ret = PyList_New(seq_length);
    for (int i = 0; i < seq_length; i++) {
        PyList_SetItem(ret, i, Py_BuildValue("i", seq[i]));
    }
    free(seq);
    return ret;
}

static PyObject *
DPInfo_get_parent(DPInfoObject *self, void *closure)
{
    if (self->dp_info->parent == NULL) {
        Py_RETURN_NONE;
    }
    DPInfoObject *ret = (DPInfoObject *)DPInfo_new(&DPInfoType, NULL, NULL);
    ret->dp_info = self->dp_info->parent;
    return (PyObject *)ret;
}

static PyGetSetDef DPInfo_getsetters[] = {
    {"cost", (getter)DPInfo_get_cost, (setter)DPInfo_set_cost, "cost", NULL},
    {"latest_insert_idx", (getter)DPInfo_get_latest_insert_idx, NULL, "latest_insert_idx", NULL},
    {"sequence", (getter)DPInfo_get_sequence, NULL, "sequence", NULL},
    {"parent", (getter)DPInfo_get_parent, NULL, "parent", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef DPInfo_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
DPInfo_setParent(DPInfoObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"parent", NULL};
    DPInfoObject *parent;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &parent))
        return NULL;
    self->dp_info->setParent(self->dp_info, parent->dp_info);
    Py_RETURN_NONE;
}

static PyMethodDef DPInfo_methods[] = {
    {"setParent", (PyCFunction)DPInfo_setParent, METH_VARARGS | METH_KEYWORDS,
    "setParent(parent: dp_info.DPInfo)\n"},
    {NULL}  /* Sentinel */
};

static PyTypeObject DPInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dp_info.DPInfo",
    .tp_basicsize = sizeof(DPInfoObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)DPInfo_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "DPInfo objects\n"
              "DPInfo(latest_insert_idx: int)\n\n",
    .tp_methods = DPInfo_methods,
    .tp_members = DPInfo_members,
    .tp_getset = DPInfo_getsetters,
    .tp_init = (initproc)DPInfo_init,
    .tp_new = DPInfo_new,
};

static PyModuleDef DPInfoModule = {
    PyModuleDef_HEAD_INIT,
    "dp_info",
    "Dynamic Programming Node Info",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_dp_info(void)
{
    PyObject *m;
    if (PyType_Ready(&DPInfoType) < 0)
        return NULL;

    m = PyModule_Create(&DPInfoModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&DPInfoType);
    PyModule_AddObject(m, "DPInfo", (PyObject *) &DPInfoType);
    return m;
}

// 封装结束
