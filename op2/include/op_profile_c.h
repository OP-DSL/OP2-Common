#ifndef __OP_PROFILE_C_H
#define __OP_PROFILE_C_H

#ifdef __cplusplus
extern "C" {
#endif

void op_profile_start(const char* name);

void op_profile_enter(const char* name);
void op_profile_enter_kernel(const char* name, const char* target, const char* variant);

void op_profile_next(const char* name);

void op_profile_exit(void);
void op_profile_end(void);

void op_profile_output(void);
void op_profile_output_json(const char* filename);

#ifdef __cplusplus
}
#endif

#endif
