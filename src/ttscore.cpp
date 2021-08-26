#include "../include/ttscore.h"

double inference(PyObject* pInstanceText2Speech, const char* text, const char* path, int sample_rate)
{
    PyObject* pText2SpeechCallArgList = PyTuple_New(3);
    PyObject* pText2SpeechCallArg0 = PyUnicode_FromString(text);
    PyObject* pText2SpeechCallArg1 = PyUnicode_FromString(path);
    PyObject* pText2SpeechCallArg2 = PyLong_FromLong(sample_rate);
    PyTuple_SetItem(pText2SpeechCallArgList, 0, pText2SpeechCallArg0);
    PyTuple_SetItem(pText2SpeechCallArgList, 1, pText2SpeechCallArg1);
    PyTuple_SetItem(pText2SpeechCallArgList, 2, pText2SpeechCallArg2);
    PyObject* pRes = PyObject_CallObject(pInstanceText2Speech, pText2SpeechCallArgList);
    double length = PyFloat_AS_DOUBLE(pRes);

    return length;
}

PyObject* getInstanceText2Speech(const char* config_file, const char* model_file, int use_gpu)
{
    PyObject* pInstanceText2Speech;
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append('../third/TTS-Core/src')");
    if (Py_IsInitialized())
    {
        PyObject* pModule = PyImport_ImportModule("Text2Speech");
        PyObject* pModuleDict = PyModule_GetDict(pModule);
        PyObject* pClassText2Speech = PyDict_GetItemString(pModuleDict, "Text2Speech");
        PyObject* pText2SpeechArgList = PyTuple_New(3);
        PyObject* pText2SpeechArg0 = PyUnicode_FromString(config_file);
        PyObject* pText2SpeechArg1 = PyUnicode_FromString(model_file);
        PyObject* pText2SpeechArg2 = PyBool_FromLong(use_gpu);
        PyTuple_SetItem(pText2SpeechArgList, 0, pText2SpeechArg0);
        PyTuple_SetItem(pText2SpeechArgList, 1, pText2SpeechArg1);
        PyTuple_SetItem(pText2SpeechArgList, 2, pText2SpeechArg2);

        PyObject* pInstanceMethodText2Speech = PyInstanceMethod_New(pClassText2Speech);
        pInstanceText2Speech = PyObject_CallObject(pInstanceMethodText2Speech, pText2SpeechArgList);
    }

    return pInstanceText2Speech;
}
