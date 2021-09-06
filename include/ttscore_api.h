#ifndef TTS_CORE_API_H
#define TTS_CORE_API_H
#include <python3.6m/Python.h>
#include "ttscore_status.h"

#ifdef __cplusplus
extern "C" { 
#endif

STATUS inference(void* pInstanceHandle, const char* text, const char* path, int sample_rate);

STATUS getInstanceHandle(void** ppInstanceHandle, const char* model_conf, const char* model_ckpt, const char* vocoder_conf, const char* vocoder_ckpt, int use_gpu);

STATUS finalize();

#ifdef __cplusplus
}
#endif

#endif