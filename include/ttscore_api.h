#ifndef TTS_CORE_API_H
#define TTS_CORE_API_H
#include <python3.8/Python.h>
#include "ttscore_status.h"

#ifdef __cplusplus
extern "C"
{
#endif

    STATUS initialize();

    STATUS inference(void *, const char *, const char *, int);

    STATUS getInstanceHandle(void **, const char *, const char *, const char *, const char *, int);

    STATUS finalize();

#ifdef __cplusplus
}
#endif

#endif