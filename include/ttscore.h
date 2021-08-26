#ifndef TTS_INTERFACE_H
#define TTS_INTERFACE_H
#include <python3.6m/Python.h>

double inference(PyObject* pInstanceText2Speech, const char* text, const char* path, int sample_rate);

PyObject* getInstanceText2Speech(const char* config_file, const char* model_file, int use_gpu);

#endif