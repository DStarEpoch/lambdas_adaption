#include "sample_generator.h"

typedef struct _PickTag {
    long lambda_idx;
    long sample_idx_at_lambda;
    double weight;
    double stop_weight;
}PickTag;

static void
insertContextInList(PyObject *all_lambdas_info, LambdaInfoContextObject *context_obj) {
    if (!PyList_Check(all_lambdas_info)) {
        PyErr_SetString(PyExc_TypeError, "all_lambdas_info must be list");
        return;
    }

    Py_ssize_t list_size = PyList_Size(all_lambdas_info);
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

// module sample_generator
static PyObject*
SampleGeneratorModule_genSamplesForInsertLambda(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"lambda_num", "samples_per_lambda", "org_u_nks",
                            "f_k", "insert_lambdas_pos", NULL};

    long lambda_num;
    long samples_per_lambda;
    PyObject *input_org_u_nks;
    PyObject *input_f_k;
    PyObject *insert_lambdas_pos;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "llOOO", kwlist, &lambda_num, &samples_per_lambda,
    &input_org_u_nks, &input_f_k, &insert_lambdas_pos)) {
        return NULL;
    }

    if (!PyList_Check(insert_lambdas_pos)) {
        PyErr_SetString(PyExc_TypeError, "insert_lambdas_pos must be list");
        return NULL;
    }

    double *org_u_nks = malloc(lambda_num * lambda_num * samples_per_lambda * sizeof(double));
    for (long i = 0; i < lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_GetItem(input_org_u_nks, i);
        for (long j = 0; j < lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_GetItem(sample_from_lambda, j);
            for (long k=0; k < samples_per_lambda; k++) {
                double potential = PyFloat_AsDouble(PyList_GetItem(eval_potential_at_lambda, k));
                org_u_nks[i * (lambda_num * samples_per_lambda) + j * samples_per_lambda + k] = potential;
            }
        }
    }

    double *f_k = malloc(lambda_num * sizeof(double));
    for (long i = 0; i < lambda_num; i++) {
        double f = PyFloat_AsDouble(PyList_GetItem(input_f_k, i));
        f_k[i] = f;
    }

    time_t t;
    unsigned seed = (unsigned) time(&t);
    srand(seed);

    Py_ssize_t insert_lambdas_size = PyList_Size(insert_lambdas_pos);
    PyObject *all_lambdas_info = PyList_New(lambda_num);
    PickTag* *pick_tags_list = (PickTag* *)malloc(sizeof(PickTag*) * insert_lambdas_size);
//    PickTag pick_tags_list[insert_lambdas_size][samples_per_lambda];
    // init all_lambdas_info with exist lambdas
    for (long i = 0; i < lambda_num; i++) {
        LambdaInfoContextObject *op = createLambdaInfoContextObjFromArgs(i, i, 0.0, 0, i, f_k[i]);
        PyList_SetItem(all_lambdas_info, i, (PyObject *)op);
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

        if (start_lambda_idx < 0 || start_lambda_idx >= lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "start_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            start_lambda_idx, i, lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }

        if (end_lambda_idx < 0 || end_lambda_idx >= lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "end_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            end_lambda_idx, i, lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }
        // check end

        // compute probability of all samples in current inserted lambda
        // f_λ = -lnΣ_jΣ_n{ exp[-u_λ(x_jn)] / Σ_k{N_k * exp[f_k-u_k(x_jn)]} }
        double sum_p = 0.0;
        double** all_u_samples = (double* *)malloc(sizeof(double*) * lambda_num);
        // guess f_λ for avoiding overflow
        double f_insert = (1 - ratio) * f_k[start_lambda_idx] + ratio * f_k[end_lambda_idx];

        for (long j = 0; j < lambda_num; j++) {
            all_u_samples[j] = (double *)malloc(sizeof(double) * samples_per_lambda);
            for (long k = 0; k < samples_per_lambda; k ++) {
                double u_inserted = org_u_nks[j*(lambda_num*samples_per_lambda) + start_lambda_idx*samples_per_lambda + k] * (1 - ratio)
                + org_u_nks[j*(lambda_num*samples_per_lambda) + end_lambda_idx*samples_per_lambda + k] * ratio;
                double bias = 0.0;

                // p_λ(x) = 1 / (Σ_j{N_j * exp[f_j + u_λ(x) - u_j(x)]})
                for (long l = 0; l < lambda_num; l ++) {
                    double log_bias_at_sample = f_k[l] - f_insert + u_inserted -
                    org_u_nks[j*(lambda_num*samples_per_lambda) + l*samples_per_lambda + k];
                    bias += exp(log_bias_at_sample);
                }

                all_u_samples[j][k] = 1.0 / (bias * samples_per_lambda);
                sum_p += all_u_samples[j][k];
            }
        }
//        printf("sum_p: %.5lf at lambda idx: %ld\n", sum_p, i);
//        f_insert = -log(sum_p);

        LambdaInfoContextObject *op;
        // use org_idx to mark index of insert lambda
        op = (LambdaInfoContextObject *)createLambdaInfoContextObjFromArgs(start_lambda_idx, end_lambda_idx, ratio, 1, i, f_insert);
        insertContextInList(all_lambdas_info, op);

        // pick samples for inserted lambda by weight
        // p_λ(x) = exp[f_λ-u_λ(x)] / Σ_k{exp[f_k-u_k(x)]}
        pick_tags_list[i] = (PickTag*)malloc(sizeof(PickTag) * samples_per_lambda);
        for (long j = 0; j < samples_per_lambda; j++) {
            double rand_choice_weight_at = rand()*1.0 / RAND_MAX;
            long select_k = lambda_num - 1;
            long select_l = samples_per_lambda - 1;
            double select_weight = 0.0;
            double stop_weight = 0.0;
            int find_flag = 0;

            for (long k = 0; k < lambda_num; k++) {
                for (long l = 0; l < samples_per_lambda; l++) {
                    select_weight = all_u_samples[k][l] / sum_p;
                    stop_weight += select_weight;
                    if (stop_weight > rand_choice_weight_at) {
                        select_k = k;
                        select_l = l;
                        find_flag = 1;
                        break;
                    }
                }
                if (find_flag) {
                    break;
                }
            }
            pick_tags_list[i][j].stop_weight = stop_weight;
            pick_tags_list[i][j].weight = select_weight;
            pick_tags_list[i][j].lambda_idx = select_k;
            pick_tags_list[i][j].sample_idx_at_lambda = select_l;
        }

//         free all_u_samples
        for (long j = 0; j < lambda_num; j++) {
            free(all_u_samples[j]);
            all_u_samples[j] = NULL;
        }
        free(all_u_samples);
        all_u_samples = NULL;
    }

    // generate fake sampling for inserted lambda and reorder u_nks for all lambdas
    Py_ssize_t all_lambdas_info_size = PyList_Size(all_lambdas_info);
    double *bp_u_nks = (double *)malloc(all_lambdas_info_size * all_lambdas_info_size * samples_per_lambda * sizeof(double));
//    printf("malloc bp_u_nks length: %ld, size: %ld\n",
//    all_lambdas_info_size * all_lambdas_info_size * samples_per_lambda,
//    all_lambdas_info_size * all_lambdas_info_size * samples_per_lambda * sizeof(double));
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
                    for (long i = 0; i < samples_per_lambda; i++) {
                        bp_u_nks[cur_idx*(all_lambdas_info_size*samples_per_lambda) + iter_idx*samples_per_lambda + i] =
                        org_u_nks[cur_org_idx*(lambda_num*samples_per_lambda) + iter_org_idx*samples_per_lambda + i];
                    }
                } else {
                    double iter_ratio = iter_info->context->ratio;
                    long iter_start_lambda_idx = iter_info->context->start_lambda_idx;
                    long iter_end_lambda_idx = iter_info->context->end_lambda_idx;
                    for (long i = 0; i < samples_per_lambda; i++) {
                        bp_u_nks[cur_idx*(all_lambdas_info_size*samples_per_lambda) + iter_idx*samples_per_lambda + i] =
                        org_u_nks[cur_org_idx*(lambda_num*samples_per_lambda) + iter_start_lambda_idx*samples_per_lambda + i] * (1 - iter_ratio) +
                        org_u_nks[cur_org_idx*(lambda_num*samples_per_lambda) + iter_end_lambda_idx*samples_per_lambda + i] * iter_ratio;
                    }
                }

            } else {
                PickTag* cur_pick_tag_list = pick_tags_list[cur_info->context->org_idx];

                if (! iter_info->context->is_insert) {
                    long iter_org_idx = iter_info->context->org_idx;

                    for (long i = 0; i < samples_per_lambda; i++) {
                        long l_idx = cur_pick_tag_list[i].lambda_idx;
                        long s_idx = cur_pick_tag_list[i].sample_idx_at_lambda;
                        bp_u_nks[cur_idx*(all_lambdas_info_size*samples_per_lambda) + iter_idx*samples_per_lambda + i] =
                        org_u_nks[l_idx*(lambda_num*samples_per_lambda) + iter_org_idx*samples_per_lambda + s_idx];
                    }

                } else {
                    double iter_ratio = iter_info->context->ratio;
                    long iter_start_lambda_idx = iter_info->context->start_lambda_idx;
                    long iter_end_lambda_idx = iter_info->context->end_lambda_idx;

                    for (long i = 0; i < samples_per_lambda; i++) {
                        long l_idx = cur_pick_tag_list[i].lambda_idx;
                        long s_idx = cur_pick_tag_list[i].sample_idx_at_lambda;
                        bp_u_nks[cur_idx*(all_lambdas_info_size*samples_per_lambda) + iter_idx*samples_per_lambda + i] =
                        org_u_nks[l_idx*(lambda_num*samples_per_lambda) + iter_start_lambda_idx*samples_per_lambda + s_idx] * (1 - iter_ratio) +
                        org_u_nks[l_idx*(lambda_num*samples_per_lambda) + iter_end_lambda_idx*samples_per_lambda + s_idx] * iter_ratio;
                    }
                }

            }

        }
    }

    // free pick_tags_list
    for (long i = 0; i < insert_lambdas_size; i++) {
        free(pick_tags_list[i]);
        pick_tags_list[i] = NULL;
    }
    free(pick_tags_list);
    pick_tags_list = NULL;

    // bp_u_nks to PyList
    PyObject *bp_u_nks_list = PyList_New(all_lambdas_info_size);
    PyObject *temp;
    for (long i = 0; i < all_lambdas_info_size; i++) {
        PyObject *op1 = PyList_New(all_lambdas_info_size);
        for (long j = 0; j < all_lambdas_info_size; j++) {
            PyObject *op2 = PyList_New(samples_per_lambda);
            for (long k = 0; k < samples_per_lambda; k++) {
                temp = PyFloat_FromDouble(bp_u_nks[i*(all_lambdas_info_size*samples_per_lambda) + j*samples_per_lambda + k]);
                PyList_SET_ITEM(op2, k, temp);
            }
            PyList_SET_ITEM(op1, j, op2);
        }
        PyList_SET_ITEM(bp_u_nks_list, i, op1);
    }

    // free bp_u_nks
    free(bp_u_nks);
    bp_u_nks = NULL;

    Py_XDECREF(insert_lambdas_pos);
    Py_XDECREF(input_org_u_nks);
    Py_XDECREF(input_f_k);
    return Py_BuildValue("OO", bp_u_nks_list, all_lambdas_info);
}

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
    {"genSamplesForInsertLambda", (PyCFunction)SampleGeneratorModule_genSamplesForInsertLambda, METH_VARARGS | METH_KEYWORDS,
    "genSamplesForInsertLambda(lambda_num: int, samples_per_lambda: int, "
    "org_u_nks: list[list[list[float]]], f_k: list[float], "
    "insert_lambdas_pos: list[Tuple(int, int, float)])\n"},
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

    Py_INCREF(&LambdaInfoContextObjectType);
    PyModule_AddObject(m, "LambdaInfoContext", (PyObject *) &LambdaInfoContextObjectType);
    return m;
}
