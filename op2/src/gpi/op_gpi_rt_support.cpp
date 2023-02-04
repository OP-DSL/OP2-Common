#include <GASPI.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

#include <op_lib_gpi.h>
#include <op_lib_mpi.h>
#include <op_mpi_core.h>

#include <op_rt_support.h> /* Check this- likely not needed.*/

#include <op_gpi_core.h> 
 

//TODO remove once gpi_utils.h merged
#define GPI_FAIL(...) (                 \
    {                                   \
        fprintf(stderr, __VA_ARGS__);   \
        exit(1);                        \
    }                                   \
    )



/* GPI reimplementation of op_exchange_halo originally found in op_mpi_rt_support.cpp 
 * IS_COMMON 
 * Lots of this is common, so can be put there. 
*/
void op_gpi_exchange_halo(op_arg *arg, int exec_flag){
    op_dat dat = arg->dat;

    //If it's not in use, don't bother!
    if(arg->opt ==0)
        return;

    //Check if arg already sent
    if(arg->sent ==1){
        GPI_FAIL("Error: halo exchange already in flight for dat %s\n",dat->name);
    }

    // For a directly accessed op_dat do not do halo exchanges if not executing
    // over
    // redundant compute block
    if(exec_flag == 0 && arg->idx == -1)
        return;

    arg->sent =0; //reset flag (TODO seems unneccesary but keep anyway)

    // need to exchange both direct and indirect data sets if they're dirty
    // return if not R/RW or if not dirty.
    if(!(arg->acc == OP_READ || arg->acc == OP_RW) 
        || (dat->dirtybit !=1))
            return;
        
    //Grab the halo lists
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];


    //-------first exchange exec elements related to this data array--------

    //sanity checks
    if (compare_sets(imp_exec_list->set, dat->set) == 0 ){
        GPI_FAIL("Import list and set mismatch\n");
    }
    if (compare_sets(exp_exec_list->set, dat->set) == 0 ){
        GPI_FAIL("Export list and set mismatch\n");
    }

    int set_elem_index;
    for (int i = 0; i < exp_exec_list->ranks_size; i++) {
      for (int j = 0; j < exp_exec_list->sizes[i]; j++) {
        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i] + j];
        //TODO time to do some thinking here...

        //memcpy into eeh segment appropriately
      }
      int gpi_rank;
      gaspi_proc_rank((gaspi_rank_t*)&gpi_rank);
      //TODO
      //gaspi_write_notify();
    }

}

/* Wait for a single arg
 * equivalent to op_mpi_waitall function
 * definitey NOT_COMMON
 */
void op_gpi_waitall(op_arg *arg){
    //Check failure conditions
    if(!(arg->opt && arg->argtype == OP_ARG_DAT && arg->sent ==1))
        return;
    

    op_dat dat = arg->dat;

    op_gpi_buffer *buff = (op_gpi_buffer*)dat->gpi_buffer;

    op_gpi_recv_obj *exec_recv_objs = buff->exec_recv_objs;
    op_gpi_recv_obj *nonexec_recv_objs = buff->nonexec_recv_objs;


    /* Stores the op_dat_index */
    gaspi_notification_id_t notif_id;
    gaspi_notification_t    notif_value;

    int recv_rank, recv_dat_index;

    /* Receive for exec elements*/
    for(int i=0;i<buff->exec_recv_count;i++){
        //GPI_SAFE( );
        gaspi_notify_waitsome(IEH_SEGMENT_ID, 
                            dat->index,
                            1,
                            &notif_id, /* Notification id should be the dat index*/
                            GPI_TIMEOUT);   

        //store and reset notification value
        gaspi_notify_reset(IEH_SEGMENT_ID,
                            notif_id,
                            &notif_value);

        recv_rank = (int) notif_value;
        recv_dat_index= (int) notif_id;

        //lookup recv object
        int obj_idx=0;
        while(obj_idx < buff->exec_recv_count && exec_recv_objs[obj_idx].remote_rank != recv_rank)
            obj_idx++;
        
        //check if it didn't find it...
        if(obj_idx >= buff->exec_recv_count)
            GPI_FAIL("Unable to find exec recv object.\n"); 

        //Use to memcpy data
        op_gpi_recv_obj *obj = &exec_recv_objs[obj_idx]; /* not neccessary but looks nicer later*/


        // Copy the data into the op_dat->data array
        memcpy(obj->memcpy_offset, (void*) obj->recv_addr, obj->size);
    }
    

    /* Receive for nonexec elements*/
    for(int i=0;i<buff->nonexec_recv_count;i++){
        //GPI_SAFE_QUEUE( );
        gaspi_notify_waitsome(INH_SEGMENT_ID, 
                            dat->index,
                            1,
                            &notif_id, /* Notification id should be the dat index*/
                            GPI_TIMEOUT);   

        //store and reset notification value
        gaspi_notify_reset(INH_SEGMENT_ID,
                            notif_id,
                            &notif_value);

        recv_rank = (int) notif_value;
        recv_dat_index= (int) notif_id;

        //lookup recv object
        int obj_idx=0;
        while(obj_idx < buff->nonexec_recv_count && nonexec_recv_objs[obj_idx].remote_rank != recv_rank)
            obj_idx++;
        
        //check if it didn't find it...
        if(obj_idx >= buff->exec_recv_count)
            GPI_FAIL("Unable to find nonexec recv object.\n"); 

        //Use to memcpy data
        op_gpi_recv_obj *obj = &nonexec_recv_objs[obj_idx]; /* not neccessary but looks nicer later*/
        
        // Copy the data into the op_dat->data array
        memcpy(obj->memcpy_offset, (void*) obj->recv_addr, obj->size);
    }


    //Do partial halo stuff
    if(arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]){
        GPI_FAIL("Not implemented partial exchange\n");
        /*
        halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
        int init = OP_export_nonexec_permap[arg->map->index]->size;
        char *buffer =
            &((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec[init * dat->size];
        for (int i = 0; i < imp_nonexec_list->size; i++) {
            int set_elem_index = imp_nonexec_list->list[i];
            memcpy((void *)&dat->data[dat->size * (set_elem_index)],
                &buffer[i * dat->size], dat->size);
        }
        */
    }

}

void op_gpi_exchange_halo_partial(op_arg *arg, int exec_flag){
    GPI_FAIL("Function is not implemented\n");
}