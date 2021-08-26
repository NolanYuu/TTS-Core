#ifndef TTS_CORE_H
#define TTS_CORE_H
#include <python3.6m/Python.h>

#ifdef __cplusplus
extern "C" { 
#endif

double inference(PyObject* pInstanceText2Speech, const char* text, const char* path, int sample_rate);

PyObject* getInstanceText2Speech(const char* config_file, const char* model_file, int use_gpu);

#ifdef __cplusplus
}
#endif

#endif