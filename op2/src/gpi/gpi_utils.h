/* Header file for GPI helper functionality */
#pragma once

#include <GASPI.h>
#include <signal.h>
#include <mpi.h>
#include <stdlib.h>

#define LOCKSTEP(rank, ...) {\
    GPI_SAFE( gaspi_barrier(OP_GPI_GLOBAL, GPI_TIMEOUT) )\
    if(rank==0) printf(__VA_ARGS__);\
}


#define GPI_TIMEOUT 4000 /* 2 Seconds*/

#define GPI_TIMEOUT_EXTRA_TRIES 1


#define ALL_MPI_PRINT(...) {\
    MPI_Barrier(MPI_COMM_WORLD);\
    printf(__VA_ARGS__);\
    fflush(stdout);\
    MPI_Barrier(MPI_COMM_WORLD);\
}


#define GPI_QUEUE_SAFE(f, queue) {\
    gaspi_rank_t _rank;\
    gaspi_proc_rank(&_rank);\
    gaspi_return_t _ret;\
    while ((_ret = (f)) == GASPI_QUEUE_FULL) {\
        fprintf(stderr,"Waiting for queue...\n");\
        gaspi_return_t _waitret = gaspi_wait ((queue), GASPI_BLOCK);\
        if( _waitret != GASPI_SUCCESS) {\
            fprintf(stderr, "On rank %d: queue was full and something went wrong with waiting at function %s at %s (%d).\n", _rank, #f, __FILE__, __LINE__);\
            fflush(stderr);\
            gaspi_proc_term(GASPI_BLOCK);\
            MPI_Abort(MPI_COMM_WORLD, 1);\
        }\
    }\
    switch (_ret) {\
    case GASPI_TIMEOUT:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) timed out.\n",_rank, #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_ERROR:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) returned GASPI_ERROR.\n",_rank, #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_SUCCESS:\
        break;\
    default:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) has not returned a GASPI return value: (%d). Are you sure it's a GASPI function?\n", _rank, #f, __FILE__, __LINE__, _ret);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    }\
}

#define GPI_SAFE(f) {\
    gaspi_rank_t _rank;\
    gaspi_proc_rank(&_rank);\
    gaspi_return_t _ret = (f);\
    switch (_ret) {\
    case GASPI_TIMEOUT:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) timed out.\n", _rank, #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_ERROR:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) returned GASPI_ERROR.\n", _rank, #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_SUCCESS:\
        break;\
    default:\
        fprintf(stderr, "On rank %d: function %s at %s (%d) has not returned a GASPI return value. You sure it's a GASPI function?\n",_rank, #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    }\
}

#define GPI_FAIL(...) {                                   \
        gaspi_rank_t _rank;\
        gaspi_proc_rank(&_rank);\
        fprintf(stderr, "On rank %d: fail at %s (%d).\n",_rank, __FILE__, __LINE__);\
        fprintf(stderr, __VA_ARGS__);   \
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);                        \
        MPI_Abort(MPI_COMM_WORLD, 1);\
    }                                   \

