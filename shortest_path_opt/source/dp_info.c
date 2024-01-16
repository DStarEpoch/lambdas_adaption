#include "dp_info.h"

long getDPInfoSequenceLength(DPInfo *self) {
    if (self == NULL)
        return 0;
    long length = getDPInfoSequenceLength(self->parent) + 1;
    return length;
}

long *getDPInfoSequence(DPInfo *self) {
    if (self == NULL)
        return NULL;
    long seq_length = getDPInfoSequenceLength(self);
    long *ret_list = (long *)malloc(sizeof(long) * seq_length);
    if (self->parent == NULL) {
        ret_list[0] = self->latest_insert_idx;
        return ret_list;
    }
    long *parent_seq = getDPInfoSequence(self->parent);
    for (long i = 0; i < seq_length - 1; i++) {
        ret_list[i] = parent_seq[i];
    }
    ret_list[seq_length - 1] = self->latest_insert_idx;
    return ret_list;
}

void setDPInfoParent(DPInfo *self, DPInfo *parent) {
    self->parent = parent;
}

void initDPInfo(DPInfo *self, long latest_insert_idx) {
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
    free(self);
    self = NULL;
}

DPInfo *newDPInfo(long latest_insert_idx) {
    DPInfo *ret = (DPInfo *)malloc(sizeof(DPInfo));
    initDPInfo(ret, latest_insert_idx);
    return ret;
}

// 实现封装DPInfo为Python对象方法
static void
DPInfo_dealloc(DPInfoObject *self)
{
    if (self->dp_info) {
        freeDPInfo(self->dp_info);
        self->dp_info = NULL;
    }
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
    long latest_insert_idx;
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
    if (PyFloat_Check(value)) {
        self->dp_info->cost = PyFloat_AsDouble(value);
        return 0;
    }

    if (PyLong_Check(value)) {
        self->dp_info->cost = PyLong_AsDouble(value);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "cost must be set as a float or int");
    return NULL;
}

static PyObject *
DPInfo_get_latest_insert_idx(DPInfoObject *self, void *closure)
{
    return Py_BuildValue("i", self->dp_info->latest_insert_idx);
}

static PyObject *
DPInfo_get_sequence(DPInfoObject *self, void *closure)
{
    long *seq = self->dp_info->getSequence(self->dp_info);
    long seq_length = self->dp_info->getSequenceLength(self->dp_info);
    PyObject *ret = PyList_New(seq_length);
    for (long i = 0; i < seq_length; i++) {
        PyList_SetItem(ret, i, Py_BuildValue("i", seq[i]));
    }
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
    "setParent(parent: dp_optimizer.DPInfo)\n"},
    {NULL}  /* Sentinel */
};

PyTypeObject DPInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dp_optimizer.DPInfo",
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
// 封装结束
