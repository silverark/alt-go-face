// +build gocv

#pragma once
#ifndef GOCV_H
#define GOCV_H

#include "wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

facesret* facerec_detect_mat(facerec* rec,  image_pointer *p, const void *mat,int type);

#ifdef __cplusplus
}
#endif

#endif /* GOCV_H */