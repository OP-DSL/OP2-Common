//
//  dotR.h
//  op2_mf
//
//  Created by Istvan Reguly on 10/1/11.
//  Copyright 2011 Reguly. All rights reserved.
//

#ifndef op2_mf_zero_res_h
#define op2_mf_zero_res_h

inline void zero_res(double *r, double *c){
  *c += (*r)*(*r);
}

#endif
