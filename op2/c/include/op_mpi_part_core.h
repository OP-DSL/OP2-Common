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

#ifndef __OP_MPI_PART_CORE_H
#define __OP_MPI_PART_CORE_H

/*
 * op_mpi_part_core.h
 *
 * Headder file for the OP2 Distributed memory (MPI) Partitioning wrapper routines,
 * data migration and support utility functions
 *
 * written by: Gihan R. Mudalige, (Started 07-04-2011)
 */

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
* Random partitioning wrapper prototype
*******************************************************************************/

void op_partition_random(op_set primary_set);

#ifdef HAVE_PARMETIS

/*******************************************************************************
* ParMetis wrapper prototypes
*******************************************************************************/

void op_partition_geom(op_dat coords);

void op_partition_kway(op_map primary_map);

void op_partition_geomkway(op_dat coords, op_map primary_map);

void op_partition_meshkway(op_map primary_map);

#endif

#ifdef HAVE_PTSCOTCH

/*******************************************************************************
* PT-SCOTCH wrapper prototypes
*******************************************************************************/

void op_partition_ptscotch(op_map primary_map);

#endif

/*******************************************************************************
* Other partitioning related routine prototypes
*******************************************************************************/

void op_partition_reverse();

#ifdef __cplusplus
}
#endif

#endif /* __OP_MPI_PART_CORE_H */

