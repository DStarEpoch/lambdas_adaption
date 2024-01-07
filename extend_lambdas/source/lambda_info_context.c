#include "lambda_info_context.h"
#include "string.h"

double getRank(LambdaInfoContext *self) {
    if (self == NULL)
        return 0.0;
    return self->start_lambda_idx * (1 - self->ratio) + self->end_lambda_idx * self->ratio;
}

char* getStr(LambdaInfoContext *self) {
    static char name[10];
    strcpy(name, "1111test");
    return name;
}

void initLambdaInfoContext(LambdaInfoContext *self, long start_lambda_idx, long end_lambda_idx, double ratio, int is_insert, long org_idx, double f_k) {
    self->f_k = f_k;
    self->is_insert = is_insert;
    self->org_idx = org_idx;
    self->start_lambda_idx = start_lambda_idx;
    self->end_lambda_idx = end_lambda_idx;
    self->ratio = ratio;

    self->getStr = getStr;
    self->getRank = getRank;
}

void freeLambdaInfoContext(LambdaInfoContext *self) {
    if (self == NULL)
        return;
    free(self);
}

LambdaInfoContext* newLambdaInfoContext(long start_lambda_idx, long end_lambda_idx, double ratio, int is_insert, long org_idx, double f_k) {
    LambdaInfoContext *obj = (LambdaInfoContext *)malloc(sizeof(LambdaInfoContext));
    initLambdaInfoContext(obj, start_lambda_idx, end_lambda_idx, ratio, is_insert, org_idx, f_k);
    return obj;
}

//
static void
LambdaInfoContext_dealloc(LambdaInfoContextObject *self)
{
    freeLambdaInfoContext(self->context);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
LambdaInfoContext_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LambdaInfoContextObject *self;
    self = (LambdaInfoContextObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->context = NULL;
    }
    return (PyObject *) self;
}

static int
LambdaInfoContext_init(LambdaInfoContextObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"start_lambda_idx", "end_lambda_idx", "ratio", "is_insert", "org_idx", "f_k", NULL};
    long start_lambda_idx;
    long end_lambda_idx;
    double ratio;

    int is_insert = 0;
    long org_idx = -1;
    double f_k = 0.0;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "lld|pld", kwlist,
                                      &start_lambda_idx, &end_lambda_idx,
                                      &ratio, &is_insert, &org_idx, &f_k))
        return -1;
    self->context = newLambdaInfoContext(start_lambda_idx, end_lambda_idx, ratio, is_insert, org_idx, f_k);
    return 0;
}

static PyObject *
LambdaInfoContext_get_rank(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("d", self->context->getRank(self->context));
}

static PyObject *
LambdaInfoContext_str(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("s", self->context->getStr(self->context));
}

static PyObject *
LambdaInfoContext_get_f_k(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("d", self->context->f_k);
}

static PyObject *
LambdaInfoContext_set_f_k(LambdaInfoContextObject *self, PyObject *value, void *closure)
{
    if (PyFloat_Check(value)) {
        self->context->f_k = PyFloat_AsDouble(value);
        return 0;
    }

    if (PyLong_Check(value)) {
        self->context->f_k = PyLong_AsDouble(value);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "f_k must be set as a float or int");
    return NULL;
}

static PyObject *
LambdaInfoContext_get_is_insert(LambdaInfoContextObject *self, void *closure)
{
    if (self->context->is_insert) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
LambdaInfoContext_get_org_idx(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("l", self->context->org_idx);
}

static PyObject *
LambdaInfoContext_get_start_lambda_idx(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("l", self->context->start_lambda_idx);
}

static PyObject *
LambdaInfoContext_get_end_lambda_idx(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("l", self->context->end_lambda_idx);
}

static PyObject *
LambdaInfoContext_get_ratio(LambdaInfoContextObject *self, void *closure)
{
    return Py_BuildValue("d", self->context->ratio);
}

static PyObject *
LambdaInfoContext_get_properties(LambdaInfoContextObject *self, void *closure)
{
    PyObject *list = PyList_New(6);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for distance_matrix");
        return NULL;
    }

    PyList_SET_ITEM(list, 0, LambdaInfoContext_get_start_lambda_idx(self, NULL));
    PyList_SET_ITEM(list, 1, LambdaInfoContext_get_end_lambda_idx(self, NULL));
    PyList_SET_ITEM(list, 2, LambdaInfoContext_get_ratio(self, NULL));

    PyList_SET_ITEM(list, 3, LambdaInfoContext_get_is_insert(self, NULL));
    PyList_SET_ITEM(list, 4, LambdaInfoContext_get_org_idx(self, NULL));
    PyList_SET_ITEM(list, 5, LambdaInfoContext_get_f_k(self, NULL));

    return Py_BuildValue("O", list);
}

static PyGetSetDef LambdaInfoContext_getsetters[] = {
    {"start_lambda_idx", (getter)LambdaInfoContext_get_start_lambda_idx, NULL, "start_lambda_idx", NULL},
    {"end_lambda_idx", (getter)LambdaInfoContext_get_end_lambda_idx, NULL, "end_lambda_idx", NULL},
    {"ratio", (getter)LambdaInfoContext_get_ratio, NULL, "ratio", NULL},

    {"is_insert", (getter)LambdaInfoContext_get_is_insert, NULL, "is_insert", NULL},
    {"org_idx", (getter)LambdaInfoContext_get_org_idx, NULL, "org_idx", NULL},
    {"f_k", (getter)LambdaInfoContext_get_f_k, (setter)LambdaInfoContext_set_f_k, "f_k", NULL},

    {"rank", (getter)LambdaInfoContext_get_rank, NULL, "rank", NULL},

    {"properties", (getter)LambdaInfoContext_get_properties, NULL, "properties", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef LambdaInfoContext_members[] = {
    {NULL}  /* Sentinel */
};

PyObject *
LambdaInfoContext_createFromProperties(PyTypeObject *cls, PyObject *args)
{
    // classmethod
    PyObject *properties;

    if (! PyArg_ParseTuple(args, "O", &properties))
        return NULL;

    if (! PyList_Check(properties)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    if (PyList_Size(properties) != 6) {
        PyErr_SetString(PyExc_ValueError, "Argument must be a list of length 6");
        return NULL;
    }

    LambdaInfoContextObject *self = (LambdaInfoContextObject *)PyObject_CallObject((PyObject *)cls, PyList_AsTuple(properties));

    if (self == NULL) {
        return NULL;
    }

    return (PyObject *)self;
}

static PyMethodDef LambdaInfoContext_methods[] = {
    {"createFromProperties", (PyCFunction)LambdaInfoContext_createFromProperties, METH_CLASS | METH_VARARGS,
     "Create a LambdaInfoContext instance from properties"},
    {NULL}  /* Sentinel */
};

PyTypeObject LambdaInfoContextObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sample_generator.LambdaInfoContext",
    .tp_str = (reprfunc)LambdaInfoContext_str,
    .tp_basicsize = sizeof(LambdaInfoContextObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)LambdaInfoContext_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "LambdaInfoContext objects\n"
              "LambdaInfoContext(start_lambda_idx: int, end_lambda_idx: int, ratio: float, is_insert: bool, org_idx: int, f_k: float)\n\n",
    .tp_methods = LambdaInfoContext_methods,
    .tp_members = LambdaInfoContext_members,
    .tp_getset = LambdaInfoContext_getsetters,
    .tp_init = (initproc)LambdaInfoContext_init,
    .tp_new = LambdaInfoContext_new,
};

