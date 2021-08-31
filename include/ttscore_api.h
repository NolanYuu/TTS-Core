#ifndef TTS_CORE_API_H
#define TTS_CORE_API_H
#include <python3.6m/Python.h>

#ifdef __cplusplus
extern "C" { 
#endif

double inference(PyObject* pInstanceText2Speech, const char* text, const char* path, int sample_rate);

PyObject* getInstanceText2Speech(const char* model_conf, const char* model_ckpt, const char* vocoder_conf, const char* vocoder_ckpt, int use_gpu);

#ifdef __cplusplus
}
#endif

#endif