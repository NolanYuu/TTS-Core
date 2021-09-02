#include "ttscore_api.h"
#include "ttscore_status.h"
#include <iostream>

STATUS inference(void* pInstanceHandle, const char* text, const char* path, int sample_rate)
{
    PyObject* pInstanceText2Speech = static_cast<PyObject*>(pInstanceHandle);
    PyObject* pText2SpeechCallArgList = PyTuple_New(3);
    PyTuple_SetItem(pText2SpeechCallArgList, 0, PyUnicode_FromString(text));
    PyTuple_SetItem(pText2SpeechCallArgList, 1, PyUnicode_FromString(path));
    PyTuple_SetItem(pText2SpeechCallArgList, 2, PyLong_FromLong(sample_rate));

    std::cout << "c++ok" << " " << pInstanceHandle << std::endl;
    PyObject_CallObject(pInstanceText2Speech, pText2SpeechCallArgList);
    std::cout << "c++ok" << " " << pInstanceText2Speech << std::endl;

    return SUCCESS;
}

STATUS getInstanceHandle(void** ppInstanceHandle, const char* model_conf, const char* model_ckpt, const char* vocoder_conf, const char* vocoder_ckpt, int use_gpu)
{
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append('../submodules/TTS-Core/src/python_api')");
    if (Py_IsInitialized())
    {
        PyObject* pModule = PyImport_ImportModule("Text2Speech");
        PyObject* pModuleDict = PyModule_GetDict(pModule);
        PyObject* pClassText2Speech = PyDict_GetItemString(pModuleDict, "Text2Speech");
        PyObject* pText2SpeechArgList = PyTuple_New(5);
        PyTuple_SetItem(pText2SpeechArgList, 0, PyUnicode_FromString(model_conf));
        PyTuple_SetItem(pText2SpeechArgList, 1, PyUnicode_FromString(model_ckpt));
        PyTuple_SetItem(pText2SpeechArgList, 2, PyUnicode_FromString(vocoder_conf));
        PyTuple_SetItem(pText2SpeechArgList, 3, PyUnicode_FromString(vocoder_ckpt));
        PyTuple_SetItem(pText2SpeechArgList, 4, PyBool_FromLong(use_gpu));

        PyObject* pInstanceMethodText2Speech = PyInstanceMethod_New(pClassText2Speech);
        PyObject* pInstanceText2Speech = PyObject_CallObject(pInstanceMethodText2Speech, pText2SpeechArgList);
        *ppInstanceHandle = static_cast<void *>(pInstanceText2Speech);

        // Py_Finalize();
        return SUCCESS;
    }
    else
    {
        return ERROR_PYINIT;
    }
}
