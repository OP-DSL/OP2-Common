#include "volna_common.h"

void OutputTime(TimerParams *timer) {
  op_printf("Iteration: %d, time: %f \n", (*timer).iter, (*timer).t);
}

