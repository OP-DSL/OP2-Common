#include <GASPI.h>

#include "op_mpi_core.h" //Include the mpi stuff as it's the bigger thing


#define EEH_SEGMENT_ID 1
#define ENH_SEGMENT_ID 2
#define IEH_SEGMENT_ID 3
#define INH_SEGMENT_ID 4


//TODO CHANGE VALUE
#define GPI_TIMEOUT 1500 /* Standard GPI timeout for timing functions*/


/* Struct storing information regarding the expected dat elements from who, where, and where to copy to */
typedef struct{
    gaspi_rank_t        remote_rank; /* Rank receiving from - (used to identify the struct) */
    gaspi_offset_t      recv_addr; /* Segment address for the receiving information (where the remote write landed) */
    char*               memcpy_offset; /* Where to memcpy the received data to. I.e. the offest from the exec/non-exec sections of the import segments*/
    int                 size; /* Number of bytes */
/*?smart linked list entry struct?*/
} op_gpi_recv_obj; 

typedef struct{
    int exec_recv_count; /* Number of recieves for import execute segment expect (i.e. number of remote ranks)*/
    int nonexec_recv_count; /* Number of recieves for import non-execute segment expect (i.e number of remote ranks)*/
    op_gpi_recv_obj *exec_recv_objs; /*  For exec elements of this dat, one for each of the expected notifications*/
    op_gpi_recv_obj *nonexec_recv_objs; /* For nonexec elements of this dat , one for each of the expected notifications*/
    int exec_min_rank;
    int nonexec_min_rank;
}op_gpi_buffer;