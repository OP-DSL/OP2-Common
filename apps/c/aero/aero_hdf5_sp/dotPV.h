//
//  dotPV.h
//  op2_mf
//
//  Created by Istvan Reguly on 10/1/11.
//  Copyright 2011 Reguly. All rights reserved.
//

#ifndef op2_mf_dotPV_h
#define op2_mf_dotPV_h
inline void dotPV(const float *p, const float *v, float *c) { *c += (*p) * (*v); }

#endif
