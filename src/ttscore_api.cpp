#include "ttscore_api.h"
#include "ttscore_status.h"

STATUS initialize()
{
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append('../submodules/TTS-Core/src/python_api')");

    if (Py_IsInitialized())
    {
        return SUCCESS;
    }
    else
    {
        return ERROR_PYINIT;
    }
}

STATUS inference(void *p_handle, const char *text, const char *path, int sample_rate)
{
    PyObject *p_instance = static_cast<PyObject *>(p_handle);
    PyObject *p_arg_list = PyTuple_New(3);
    PyTuple_SetItem(p_arg_list, 0, PyUnicode_FromString(text));
    PyTuple_SetItem(p_arg_list, 1, PyUnicode_FromString(path));
    PyTuple_SetItem(p_arg_list, 2, PyLong_FromLong(sample_rate));

    PyObject_CallObject(p_instance, p_arg_list);

    return SUCCESS;
}

STATUS getInstanceHandle(void **pp_handle, const char *model_conf, const char *model_ckpt, const char *vocoder_conf, const char *vocoder_ckpt, int use_gpu)
{
    PyObject *p_module = PyImport_ImportModule("text2speech");
    PyObject *p_module_dict = PyModule_GetDict(p_module);
    PyObject *p_class = PyDict_GetItemString(p_module_dict, "Text2Speech");
    PyObject *p_arg_list = PyTuple_New(5);
    PyTuple_SetItem(p_arg_list, 0, PyUnicode_FromString(model_conf));
    PyTuple_SetItem(p_arg_list, 1, PyUnicode_FromString(model_ckpt));
    PyTuple_SetItem(p_arg_list, 2, PyUnicode_FromString(vocoder_conf));
    PyTuple_SetItem(p_arg_list, 3, PyUnicode_FromString(vocoder_ckpt));
    PyTuple_SetItem(p_arg_list, 4, PyBool_FromLong(use_gpu));

    PyObject *p_instance_method = PyInstanceMethod_New(p_class);
    PyObject *p_instance = PyObject_CallObject(p_instance_method, p_arg_list);
    *pp_handle = static_cast<void *>(p_instance);

    return SUCCESS;
}

STATUS finalize()
{
    Py_Finalize();

    return SUCCESS;
}