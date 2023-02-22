#pragma once

#include <GASPI.h>
#include <signal.h>
#include <mpi.h>
#include <stdlib.h>


#define GPI_TIMEOUT 1500


#define GPI_TIMEOUT_EXTRA_TRIES 1

#define GPI_QUEUE_SAFE(f, queue) {\
    gaspi_return_t _ret;\
    while ((_ret = (f)) == GASPI_QUEUE_FULL) {\
        gaspi_return_t _waitret = gaspi_wait ((queue), GASPI_BLOCK);\
        if( _waitret != GASPI_SUCCESS) {\
            fprintf(stderr, "Queue was full and something went wrong with waiting at function %s at %s (%d).\n", #f, __FILE__, __LINE__);\
            fflush(stderr);\
            gaspi_proc_term(GASPI_BLOCK);\
            MPI_Abort(MPI_COMM_WORLD, 1);\
        }\
    }\
    switch (_ret) {\
    case GASPI_TIMEOUT:\
        fprintf(stderr, "Function %s at %s (%d) timed out.\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_ERROR:\
        fprintf(stderr, "Function %s at %s (%d) returned GASPI_ERROR.\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_SUCCESS:\
        break;\
    default:\
        fprintf(stderr, "Function %s at %s (%d) has not returned a GASPI return value. You sure it's a GASPI function?\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    }\
}

#define GPI_SAFE(f) {\
    gaspi_return_t _ret = (f);\
    switch (_ret) {\
    case GASPI_TIMEOUT:\
        fprintf(stderr, "Function %s at %s (%d) timed out.\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_ERROR:\
        fprintf(stderr, "Function %s at %s (%d) returned GASPI_ERROR.\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    case GASPI_SUCCESS:\
        break;\
    default:\
        fprintf(stderr, "Function %s at %s (%d) has not returned a GASPI return value. You sure it's a GASPI function?\n", #f, __FILE__, __LINE__);\
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);\
        MPI_Abort(MPI_COMM_WORLD, 1);\
        break;\
    }\
}

#define GPI_FAIL(...) (                 \
    {                                   \
        fprintf(stderr, "Fail at %s (%d).\n", __FILE__, __LINE__);\
        fprintf(stderr, __VA_ARGS__);   \
        fflush(stderr);\
        gaspi_proc_term(GASPI_BLOCK);                        \
        MPI_Abort(MPI_COMM_WORLD, 1);\
    }                                   \
)

