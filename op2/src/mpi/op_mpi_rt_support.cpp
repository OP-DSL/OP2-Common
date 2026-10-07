/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * op_mpi_rt_support.c
 *
 * Support routines for the CPU-only MPI library. Halo exchanges are the unified
 * exchange's (op_mpi_unified_exchanges.cpp), on its host backend.
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

// mpi header
#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

#include <op_lib_mpi.h>
#include <op_mpi_core.h>
#include <op_rt_support.h>

void op_upload_dat(op_dat dat) { (void)dat; }

void op_download_dat(op_dat dat) {(void)dat;}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat data) {
  partition(lib_name, lib_routine, prime_set, prime_map, data);
}

void op_move_to_device() {}

void deviceSync() {}

// This library variant has no accelerator backend, so unified exchanges only
// ever run on the host one.
#include <op_mpi_unified_backend.h>

namespace op::unified_exchanges {
Backend *device_backend() { return nullptr; }
}
