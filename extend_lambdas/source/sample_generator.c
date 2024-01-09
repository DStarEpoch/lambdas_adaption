#include "sample_generator.h"

// class SampleGenerator
static void
SampleGeneratorObject_dealloc(SampleGeneratorObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
SampleGeneratorObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SampleGeneratorObject *self;
    self = (SampleGeneratorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->lambda_num = 0;
        self->samples_per_lambda = 0;
        self->org_u_nks = NULL;
        self->f_k = NULL;
    }
    return (PyObject *) self;
}

static int
SampleGeneratorObject_init(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"lambda_num", "samples_per_lambda", "org_u_nks", "f_k", NULL};

    PyObject *org_u_nks;
    PyObject *f_k;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "llOO", kwlist,
                                      &self->lambda_num, &self->samples_per_lambda,
                                      &org_u_nks, &f_k))
        return -1;

    if (!PyList_Check(org_u_nks)) {
        PyErr_SetString(PyExc_TypeError, "org_u_nks must be list");
        return -1;
    }

    if (!PyList_Check(f_k)) {
        PyErr_SetString(PyExc_TypeError, "f_k must be list");
        return -1;
    }

    self->org_u_nks = malloc(self->lambda_num * self->lambda_num * self->samples_per_lambda * sizeof(double));
    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_GetItem(org_u_nks, i);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_GetItem(sample_from_lambda, j);
            for (long k=0; k < self->samples_per_lambda; k++) {
                double potential = PyFloat_AsDouble(PyList_GetItem(eval_potential_at_lambda, k));
                self->org_u_nks[i * (self->lambda_num * self->samples_per_lambda) + j * self->samples_per_lambda + k] = potential;
            }
        }
    }

    self->f_k = malloc(self->lambda_num * sizeof(double));
    for (long i = 0; i < self->lambda_num; i++) {
        double f = PyFloat_AsDouble(PyList_GetItem(f_k, i));
        self->f_k[i] = f;
    }

    return 0;
}

static PyObject *
SampleGeneratorObject_get_org_u_nks(SampleGeneratorObject *self, void *closure)
{
    PyObject *list = PyList_New(self->lambda_num);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for org_u_nks");
        return NULL;
    }

    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_New(self->lambda_num);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_New(self->samples_per_lambda);
            for (long k = 0; k < self->samples_per_lambda; k++) {
                double potential = self->org_u_nks[i * (self->lambda_num * self->samples_per_lambda) + j * self->samples_per_lambda + k];
                PyList_SET_ITEM(eval_potential_at_lambda, k, PyFloat_FromDouble(potential));
            }
            PyList_SET_ITEM(sample_from_lambda, j, eval_potential_at_lambda);
        }
        PyList_SET_ITEM(list, i, sample_from_lambda);
    }

    return Py_BuildValue("O", list);
}

static PyObject *
SampleGeneratorObject_get_f_k(SampleGeneratorObject *self, void *closure)
{
    PyObject *list = PyList_New(self->lambda_num);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for f_k");
        return NULL;
    }

    for (long k = 0; k < self->lambda_num; k++) {
        double f = self->f_k[k];
        PyList_SET_ITEM(list, k, PyFloat_FromDouble(f));
    }

    return Py_BuildValue("O", list);
}

typedef struct _PickTag {
    long lambda_idx;
    long sample_idx_at_lambda;
    double weight;
    double stop_weight;
}PickTag;

void insertContextInList(PyObject *all_lambdas_info, LambdaInfoContextObject *context_obj) {
    if (!PyList_Check(all_lambdas_info)) {
        PyErr_SetString(PyExc_TypeError, "all_lambdas_info must be list");
        return;
    }

    long list_size = PyList_Size(all_lambdas_info);
    if (list_size < 1) {
        PyList_Append(all_lambdas_info, (PyObject *)context_obj);
        return;
    }

    LambdaInfoContextObject *last_obj = (LambdaInfoContextObject *)PyList_GetItem(all_lambdas_info, list_size - 1);
    if (getRank(last_obj->context) <= getRank(context_obj->context)) {
        PyList_Append(all_lambdas_info, (PyObject *)context_obj);
        return;
    }

    for (long i = 0; i < list_size; i++) {
        LambdaInfoContextObject *op_obj = (LambdaInfoContextObject *)PyList_GetItem(all_lambdas_info, i);
        if (getRank(context_obj->context) < getRank(op_obj->context)) {
            PyList_Insert(all_lambdas_info, i, (PyObject *)context_obj);
            return;
        }
    }
}

static PyObject *
SampleGeneratorObject_genSamplesForInsertLambda(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"insert_lambdas_pos", NULL};

    PyObject *insert_lambdas_pos;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &insert_lambdas_pos)) {
        return NULL;
    }

    if (!PyList_Check(insert_lambdas_pos)) {
        PyErr_SetString(PyExc_TypeError, "insert_lambdas_pos must be list");
        return NULL;
    }

    time_t t;
    srand((unsigned) time(&t));

    long insert_lambdas_size = PyList_Size(insert_lambdas_pos);
    PyObject *all_lambdas_info = PyList_New(self->lambda_num);
    PickTag* *pick_tags_list = (PickTag* *)malloc(sizeof(PickTag*) * insert_lambdas_size);
    // init all_lambdas_info with exist lambdas
    for (long i = 0; i < self->lambda_num; i++) {
        LambdaInfoContextObject *op;
        op = (LambdaInfoContextObject *)createLambdaInfoContextObjFromArgs(i, i, 0.0, 0, i, self->f_k[i]);
        PyList_SET_ITEM(all_lambdas_info, i, (PyObject *)op);
    }

    // pre-compute free energy of inserted lambdas and picking list
    for (long i = 0; i < insert_lambdas_size; i++) {
        long start_lambda_idx;
        long end_lambda_idx;
        double ratio;

        // check value of insert_lambdas_pos
        if (! PyArg_ParseTuple(PyList_GetItem(insert_lambdas_pos, i), "lld",
        &start_lambda_idx, &end_lambda_idx, &ratio)) {
            char err_msg[128];
            sprintf(err_msg, "insert_lambdas_pos[%ld] cannot parsed as Tuple[start_lambda_idx:int, end_lambda_idx:int, ratio:float]\n", i);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }

        if (start_lambda_idx < 0 || start_lambda_idx >= self->lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "start_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            start_lambda_idx, i, self->lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }

        if (end_lambda_idx < 0 || end_lambda_idx >= self->lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "end_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            end_lambda_idx, i, self->lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }
        // check end

        // compute probability of all samples in current inserted lambda
        // f_λ = -lnΣ_jΣ_n{ exp[-u_λ(x_jn)] / Σ_k{exp[f_k-u_k(x_jn)]} }
        double sum_p = 0.0;
        double **all_u_samples = (double **)malloc(sizeof(double *) * self->lambda_num);

        for (long j = 0; j < self->lambda_num; j++) {
            double *p_over_bias_list = (double *)malloc(sizeof(double) * self->samples_per_lambda);
            for (long k = 0; k < self->samples_per_lambda; k ++) {
                double bias = 0.0;
                for (long l = 0; l < self->lambda_num; l ++) {
                    bias += exp(self->f_k[l] - self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + l*self->samples_per_lambda + k]);
                }
                double p_over_bias = exp(-(self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + start_lambda_idx*self->samples_per_lambda + k] * (1 - ratio)
                + self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + end_lambda_idx*self->samples_per_lambda + k] * ratio));
                p_over_bias_list[k] = p_over_bias / (bias * self->samples_per_lambda);
                sum_p += p_over_bias_list[k];
            }
            all_u_samples[j] = p_over_bias_list;
        }
//        double f_insert = -log(sum_p);
        double f_insert = (1 - ratio) * self->f_k[start_lambda_idx] + ratio * self->f_k[end_lambda_idx];

        LambdaInfoContextObject *op;
        // use org_idx to mark index of insert lambda
        op = (LambdaInfoContextObject *)createLambdaInfoContextObjFromArgs(start_lambda_idx, end_lambda_idx, ratio, 1, i, f_insert);
        insertContextInList(all_lambdas_info, op);

        // pick samples for inserted lambda by weight
        // p_λ(x) = exp[f_λ-u_λ(x)] / Σ_k{exp[f_k-u_k(x)]}
        PickTag *candidate_tags = (PickTag*)malloc(sizeof(PickTag) * self->lambda_num * self->samples_per_lambda);
        double stop_weight = 0.0;
        for (long j = 0; j < self->lambda_num; j++) {
            for (long k = 0; k < self->samples_per_lambda; k ++) {
                long ptr = j * self->samples_per_lambda + k;
                stop_weight += all_u_samples[j][k] / sum_p;
                candidate_tags[ptr].stop_weight = stop_weight;
                candidate_tags[ptr].weight = all_u_samples[j][k] / sum_p;
                candidate_tags[ptr].lambda_idx = j;
                candidate_tags[ptr].sample_idx_at_lambda = k;
            }
        }

        pick_tags_list[i] = (PickTag*)malloc(sizeof(PickTag) * self->samples_per_lambda);
        for (long j = 0; j < self->samples_per_lambda; j++) {
            double rand_choice_weight_at = rand()*1.0 / RAND_MAX;
            long select_tag_idx = self->samples_per_lambda * self->lambda_num - 1;
            for (long k = 0; k < self->samples_per_lambda * self->lambda_num; k++) {
                if (candidate_tags[k].stop_weight > rand_choice_weight_at) {
                    select_tag_idx = k;
                    break;
                }
            }
            pick_tags_list[i][j].stop_weight = candidate_tags[select_tag_idx].stop_weight;
            pick_tags_list[i][j].weight = candidate_tags[select_tag_idx].weight;
            pick_tags_list[i][j].lambda_idx = candidate_tags[select_tag_idx].lambda_idx;
            pick_tags_list[i][j].sample_idx_at_lambda = candidate_tags[select_tag_idx].sample_idx_at_lambda;
        }

        // free candidate_tags
        free(candidate_tags);
        // free all_u_samples
        for (long j = 0; j < self->lambda_num; j++) {
            free(all_u_samples[j]);
        }
        free(all_u_samples);
        all_u_samples = NULL;
    }

    // generate fake sampling for inserted lambda and reorder u_nks for all lambdas
    long all_lambdas_info_size = PyList_Size(all_lambdas_info);
    double *bp_u_nks = (double *)malloc(all_lambdas_info_size * all_lambdas_info_size * self->samples_per_lambda * sizeof(double));
    for (long cur_idx = 0; cur_idx < all_lambdas_info_size; cur_idx++) {
        LambdaInfoContextObject* cur_info = (LambdaInfoContextObject *) PyList_GetItem(all_lambdas_info, cur_idx);

        for (long iter_idx = 0; iter_idx < all_lambdas_info_size; iter_idx++) {
            // l2->cur
            // l1->iter, 1->iter_start, 1'->iter_end
            // u_l2_at_l1 = u_l2_at_1 * (1-r1) + u_l2_at_1' * r1
            LambdaInfoContextObject* iter_info = (LambdaInfoContextObject *) PyList_GetItem(all_lambdas_info, iter_idx);

            if (! cur_info->context->is_insert) {
                long cur_org_idx = cur_info->context->org_idx;

                if (! iter_info->context->is_insert) {
                    long iter_org_idx = iter_info->context->org_idx;
                    for (long i = 0; i < self->samples_per_lambda; i++) {
                        bp_u_nks[cur_idx*(all_lambdas_info_size*self->samples_per_lambda) + iter_idx*self->samples_per_lambda + i] =
                        self->org_u_nks[cur_org_idx*(self->lambda_num*self->samples_per_lambda) + iter_org_idx*self->samples_per_lambda + i];
                    }
                } else {
                    double iter_ratio = iter_info->context->ratio;
                    long iter_start_lambda_idx = iter_info->context->start_lambda_idx;
                    long iter_end_lambda_idx = iter_info->context->end_lambda_idx;
                    for (long i = 0; i < self->samples_per_lambda; i++) {
                        bp_u_nks[cur_idx*(all_lambdas_info_size*self->samples_per_lambda) + iter_idx*self->samples_per_lambda + i] =
                        self->org_u_nks[cur_org_idx*(self->lambda_num*self->samples_per_lambda) + iter_start_lambda_idx*self->samples_per_lambda + i] * (1 - iter_ratio) +
                        self->org_u_nks[cur_org_idx*(self->lambda_num*self->samples_per_lambda) + iter_end_lambda_idx*self->samples_per_lambda + i] * iter_ratio;
                    }
                }

            } else {
                PickTag* cur_pick_tag_list = pick_tags_list[cur_info->context->org_idx];

                if (! iter_info->context->is_insert) {
                    long iter_org_idx = iter_info->context->org_idx;

                    for (long i = 0; i < self->samples_per_lambda; i++) {
                        long l_idx = cur_pick_tag_list[i].lambda_idx;
                        long s_idx = cur_pick_tag_list[i].sample_idx_at_lambda;
                        bp_u_nks[cur_idx*(all_lambdas_info_size*self->samples_per_lambda) + iter_idx*self->samples_per_lambda + i] =
                        self->org_u_nks[l_idx*(self->lambda_num*self->samples_per_lambda) + iter_org_idx*self->samples_per_lambda + s_idx];
                    }

                } else {
                    double iter_ratio = iter_info->context->ratio;
                    long iter_start_lambda_idx = iter_info->context->start_lambda_idx;
                    long iter_end_lambda_idx = iter_info->context->end_lambda_idx;

                    for (long i = 0; i < self->samples_per_lambda; i++) {
                        long l_idx = cur_pick_tag_list[i].lambda_idx;
                        long s_idx = cur_pick_tag_list[i].sample_idx_at_lambda;
                        bp_u_nks[cur_idx*(all_lambdas_info_size*self->samples_per_lambda) + iter_idx*self->samples_per_lambda + i] =
                        self->org_u_nks[l_idx*(self->lambda_num*self->samples_per_lambda) + iter_start_lambda_idx*self->samples_per_lambda + s_idx] * (1 - iter_ratio) +
                        self->org_u_nks[l_idx*(self->lambda_num*self->samples_per_lambda) + iter_end_lambda_idx*self->samples_per_lambda + s_idx] * iter_ratio;
                    }
                }

            }

        }
    }

    // free pick_tags_list
    for (long i = 0; i < insert_lambdas_size; i++) {
        free(pick_tags_list[i]);
    }
    free(pick_tags_list);
    pick_tags_list = NULL;

    // bp_u_nks to PyList
    PyObject *bp_u_nks_list = PyList_New(all_lambdas_info_size);
    for (long i = 0; i < all_lambdas_info_size; i++) {
        PyObject *op1 = PyList_New(all_lambdas_info_size);
        for (long j = 0; j < all_lambdas_info_size; j++) {
            PyObject *op2 = PyList_New(self->samples_per_lambda);
            for (long k = 0; k < self->samples_per_lambda; k++) {
                PyList_SET_ITEM(op2, k, PyFloat_FromDouble(bp_u_nks[i*(all_lambdas_info_size*self->samples_per_lambda) + j*self->samples_per_lambda + k]));
            }
            PyList_SET_ITEM(op1, j, op2);
        }
        PyList_SET_ITEM(bp_u_nks_list, i, op1);
    }

    // free bp_u_nks
    free(bp_u_nks);
    bp_u_nks = NULL;

    return Py_BuildValue("OO", bp_u_nks_list, all_lambdas_info);
}

static PyGetSetDef SampleGeneratorObject_getsetters[] = {
    {"org_u_nks", (getter)SampleGeneratorObject_get_org_u_nks, NULL, "org_u_nks", NULL},
    {"f_k", (getter)SampleGeneratorObject_get_f_k, NULL, "f_k", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef SampleGeneratorObject_members[] = {
    {"lambda_num", T_INT, offsetof(SampleGeneratorObject, lambda_num), 0,
     "number of lambdas"},
    {"samples_per_lambda", T_INT, offsetof(SampleGeneratorObject, samples_per_lambda), 0,
     "number of potential samples at each lambda"},
    {NULL}  /* Sentinel */
};

static PyMethodDef SampleGeneratorObject_methods[] = {
    {"genSamplesForInsertLambda",
    (PyCFunction)SampleGeneratorObject_genSamplesForInsertLambda,
    METH_VARARGS | METH_KEYWORDS,
     "genSamplesForInsertLambda(insert_lambdas_pos: list[tuple(int, int, float)]) \n"
     "-> Tuple[bp_u_nks: List[List[List[float]]], all_lambdas_info: List[LambdaInfoContext]]"},
    {NULL}  /* Sentinel */
};

PyTypeObject SampleGeneratorObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sample_generator.SampleGenerator",
    .tp_basicsize = sizeof(SampleGeneratorObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)SampleGeneratorObject_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "SampleGenerator objects\n"
              "SampleGenerator(lambda_num: int, samples_per_lambda: int, org_u_nks: list[list[list[float]]])\n\n",
    .tp_methods = SampleGeneratorObject_methods,
    .tp_members = SampleGeneratorObject_members,
    .tp_getset = SampleGeneratorObject_getsetters,
    .tp_init = (initproc)SampleGeneratorObject_init,
    .tp_new = SampleGeneratorObject_new,
};

// module sample_generator
static PyObject *
SampleGeneratorModule_copy_context(PyObject *self, PyObject *args, PyObject *kwds)
{

    PyObject *op;
    op = LambdaInfoContext_createFromProperties(&LambdaInfoContextObjectType, args);
    if (op) {
        return (PyObject *) op;
    }
    printf("failed to create LambdaInfoContext from properties\n");
    Py_RETURN_NONE;
}

static PyMethodDef SampleGeneratorModule_methods[] = {
    {"copyLambdaInfoContextFromProperties", (PyCFunction)SampleGeneratorModule_copy_context, METH_VARARGS | METH_KEYWORDS,
    "create a new LambdaInfoContext object from properties of an exist object\n"},
    {NULL}  /* Sentinel */
};

static PyModuleDef SampleGeneratorModule = {
    PyModuleDef_HEAD_INIT,
    "sample_generator",
    "Dynamic Programming optimization module",
    -1,
    SampleGeneratorModule_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_sample_generator(void)
{
    PyObject *m = PyModule_Create(&SampleGeneratorModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&LambdaInfoContextObjectType) < 0)
        return NULL;

    if (PyType_Ready(&SampleGeneratorObjectType) < 0)
        return NULL;

    Py_INCREF(&LambdaInfoContextObjectType);
    PyModule_AddObject(m, "LambdaInfoContext", (PyObject *) &LambdaInfoContextObjectType);
    Py_INCREF(&SampleGeneratorObjectType);
    PyModule_AddObject(m, "SampleGenerator", (PyObject *) &SampleGeneratorObjectType);
    return m;
}
