#include <GASPI.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

#include <op_lib_gpi.h>
#include <op_lib_mpi.h>
#include <op_mpi_core.h>

#include <op_rt_support.h> /* Check this- likely not needed.*/

#include <op_gpi_core.h> 
 
#include "gpi_utils.h"

/* returns the number of 1 bits in the character */
int bit_count_char(char byte){

    int sum=0;
    for(int i=0;i<8;i++){
        sum+=(byte&1);
        byte=byte>>1;
    }
    return sum;
}

/* returns the number of 1 bits in the */
int bit_count(void *data, int size){
    char *d = (char*) data;
    int sum=0;
    for(int c=0;c<size;c++){
        sum+=bit_count_char(d[c]);
    }

    return sum;
}


int xor_byte_summation(void *data, int size){
    char *d = (char*) data;
    char res=d[0];
    for(int i=1;i<size;i++){
        res = res ^ d[i];
    }
    return res;
}

/* GPI reimplementation of op_exchange_halo originally found in op_mpi_rt_support.cpp 
 * IS_COMMON 
 * Lots of this is common, so can be put there. 
 * TODO checks are required to ensure the offsets are correct within the segments.
*/
void op_gpi_exchange_halo(op_arg *arg, int exec_flag){
    op_dat dat = arg->dat;
    op_gpi_buffer gpi_buf = (op_gpi_buffer) dat->gpi_buffer;

    // if(dat->set->index != dat->index) {
    //     GPI_FAIL("Error: op_exchange_halo()\n",dat->name);
    // }

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

    int gpi_rank;
    gaspi_proc_rank((gaspi_rank_t*)&gpi_rank);

    //LOCKSTEP(gpi_rank,"exchanging %s dat\n",arg->dat->name);

    //-------first exchange exec elements related to this data array--------

    //sanity checks
    if (compare_sets(imp_exec_list->set, dat->set) == 0 ){
        GPI_FAIL("Import list and set mismatch\n");
    }
    if (compare_sets(exp_exec_list->set, dat->set) == 0 ){
        GPI_FAIL("Export list and set mismatch\n");
    }

    //dat offset inside eeh segment
    //Note - changed to int to not perform addition on pointer type
    // and simplifies offset logic as operations are now performed on byte count.
    void *dat_offset_addr = (void*)(eeh_segment_ptr + dat->loc_eeh_seg_off);

    int set_elem_index;
    for (int i = 0; i < exp_exec_list->ranks_size; i++) {
      for (int j = 0; j < exp_exec_list->sizes[i]; j++) {

        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i] + j];
        //Can reuse the exp_exec_list->disps[i] as this gives the per rank displacement into the dat buffer.

        

        //memcpy into eeh segment appropriately
        //TODO check the offsets are correct for the dest w.r.t bytes n stuff
        memcpy(((void*)dat_offset_addr + exp_exec_list->disps[i]* dat->size + j* dat->size),
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }


      //get remote offsets for that rank
      gaspi_offset_t remote_exec_offset = (gaspi_offset_t) gpi_buf->remote_exec_offsets[i];
      
      //printf("Dat: %s: Rank %d sending %d bytes of exec data to rank %d with remote segment offset %d\n",dat->name, gpi_rank, dat->size * exp_exec_list->sizes[i], exp_exec_list->ranks[i], remote_exec_offset);

      gaspi_offset_t local_offset = (gaspi_offset_t) dat->loc_eeh_seg_off+ exp_exec_list->disps[i]*dat->size;
      GPI_QUEUE_SAFE( gaspi_write_notify(EEH_SEGMENT_ID, /* local segment id*/
                        local_offset, /* local segment offset*/
                        exp_exec_list->ranks[i], /* remote rank*/
                        IEH_SEGMENT_ID, /* remote segment id*/
                        remote_exec_offset, /* remote offset*/
                        dat->size * exp_exec_list->sizes[i], /* send size*/
                        dat->index <<7 | gpi_rank, /* notification id*/
                        1, /* notification value, 1 bit added for non-zero notif values */
                        OP2_GPI_QUEUE_ID, /* queue id*/
                        GPI_TIMEOUT /* timeout*/
                        ), OP2_GPI_QUEUE_ID )
#ifdef GPI_VERBOSE
      printf("Rank %d sent execute %s dat data to rank %d with not_ID %d \n",gpi_rank,dat->name, exp_exec_list->ranks[i],dat->index <<7 | gpi_rank);
      fflush(stdout);
#endif
    }


    //Second exchange for nonexec elements. 
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0){
        GPI_FAIL("Error: Non-Import list and set mismatch");
    }

    if(compare_sets(exp_nonexec_list->set, dat->set) == 0){
        GPI_FAIL("Error: Non-Export list and set mismatch");
    }

    dat_offset_addr = (void*)(enh_segment_ptr + dat->loc_enh_seg_off);

    for (int i =0; i < exp_nonexec_list->ranks_size; i++){
        for (int j=0;j<exp_nonexec_list->sizes[i];j++){
            set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i] + j];

            memcpy((void*)dat_offset_addr+exp_nonexec_list->disps[i]* dat->size + j *dat->size,
                   (void*)&dat->data[dat->size * (set_elem_index)],
                    dat->size);

        }
        
        gaspi_offset_t remote_nonexec_offset = (gaspi_offset_t) gpi_buf->remote_nonexec_offsets[i];

        //fprintf(stderr,"remote offset: %ld for rank %d in segment %d\n",remote_offset, exp_nonexec_list->ranks[i], INH_SEGMENT_ID);
        //fflush(stderr);

        GPI_QUEUE_SAFE( gaspi_write_notify(
                           ENH_SEGMENT_ID, /* local segment */
                           (gaspi_offset_t) dat->loc_enh_seg_off + exp_nonexec_list->disps[i]*dat->size, /* local segment offset*/
                           exp_nonexec_list->ranks[i], /* remote rank*/
                           INH_SEGMENT_ID, /* remote segment */
                           remote_nonexec_offset, /* remote segment offset*/
                           dat->size * exp_nonexec_list->sizes[i], /* data to send (in bytes)*/
                           dat->index<<7 | gpi_rank, /* notification id*/
                           1, /* notification value. 1 added for non-zero notif values */
                           OP2_GPI_QUEUE_ID, /* queue id*/
                           GPI_TIMEOUT /* timeout */
                           ), OP2_GPI_QUEUE_ID )
      
#ifdef GPI_VERBOSE
        printf("Rank %d sent non-execute %s dat data to rank %d with not_ID %d.\n",gpi_rank,dat->name, exp_nonexec_list->ranks[i], dat->index <<7 | gpi_rank);
        fflush(stdout);
#endif
    }


    //Finish up
    dat->dirtybit =0;
    arg->sent=1;

}


/* Wait for a single arg
 * equivalent to op_mpi_waitall function
 * definitey NOT_COMMON
 */
void op_gpi_waitall(op_arg *arg){
    //Check skip conditions
    if(!(arg->opt && arg->argtype == OP_ARG_DAT && arg->sent ==1))
        return;
    

    gaspi_rank_t rank;
    gaspi_proc_rank(&rank);

    op_dat dat = arg->dat;

    op_gpi_buffer buff = (op_gpi_buffer)dat->gpi_buffer;

    op_gpi_recv_obj *exec_recv_objs = buff->exec_recv_objs;
    op_gpi_recv_obj *nonexec_recv_objs = buff->nonexec_recv_objs;


    /* Stores the op_dat_index */
    gaspi_notification_id_t notif_id;
    gaspi_notification_t    notif_value;

    int recv_rank, recv_dat_index;

#ifdef GPI_VERBOSE
    printf("Rank %d expects %d exec receives from ranks:\n",rank,buff->exec_recv_count);
    for(int i=0;i<buff->exec_recv_count;i++){
        printf("%d ",buff->exec_recv_objs[i].remote_rank);
    }
    printf("\n");
    fflush(stdout);
#endif


    /* TODO move somewhere else and store */
    int max_expected_rank=0;
    for(int i=0;i<buff->exec_recv_count;i++){
        if(buff->exec_recv_objs[i].remote_rank>max_expected_rank)
            max_expected_rank=buff->exec_recv_objs[i].remote_rank;
    }



    /* Receive for exec elements*/
    for(int i=0;i<buff->exec_recv_count;i++){
        GPI_SAFE( gaspi_notify_waitsome(IEH_SEGMENT_ID, 
                            dat->index << 7,
                            10, /* Should be max_expected rank*/
                            &notif_id, /* Notification id should be the dat index*/
                            GPI_TIMEOUT) )

#ifdef GPI_VERBOSE
        printf("Rank %d received exec not_ID %d.\n",rank,notif_id);
#endif
        
        recv_rank = notif_id & ((1<<7)-1); /* Filter out the upper half */
        recv_dat_index= (int) notif_id>> 7; /* Get only the upper half*/
        
        //Sanity check
        if(recv_dat_index!=dat->index){
            GPI_FAIL("Accepted exec notification from unexpected dat.\n Expected: %d got:%d\n",dat->index, notif_id);
        }
        
        //store and reset notification value
        GPI_SAFE( gaspi_notify_reset(IEH_SEGMENT_ID,
                            notif_id,
                            &notif_value) ) 

        
        //printf("notif_value: %d recv rank: %d\n",notif_value, recv_rank);



        //lookup recv object
        int obj_idx=0;
        while((obj_idx < buff->exec_recv_count) && (exec_recv_objs[obj_idx].remote_rank != recv_rank))
            obj_idx++;
        

        //check if it didn't find it...
        if(obj_idx >= buff->exec_recv_count){
            printf("Dumping exec recv objs:\n");
            printf("Exec recv count: %d\n",buff->exec_recv_count);
            for(int i=0;i<buff->exec_recv_count;i++){
              op_gpi_recv_obj *obj = &buff->exec_recv_objs[i];
              printf(" - remote rank: %d\n - segment_recv_offset: %d\n - memcpy addr: %p\n - size: %d\n\n",obj->remote_rank,obj->segment_recv_offset, obj->memcpy_addr, obj->size);
            }
            fflush(stdout);
            GPI_FAIL("Unable to find exec recv object from rank %d in dat %s\n",recv_rank, dat->name); 
        }


        //Use to memcpy data
        op_gpi_recv_obj *obj = &exec_recv_objs[obj_idx]; /* not neccessary but looks nicer later*/

        //printf("before memcpy to address: %p \n", (void*)(ieh_segment_ptr + obj->segment_recv_offset));
        //fflush(stdout);



        // Copy the data into the op_dat->data array
        memcpy(obj->memcpy_addr, (void*) (ieh_segment_ptr + obj->segment_recv_offset), obj->size);

        // XOR bitwise test
        //printf("Dat: %s: Rank %d received %d bytes of exec data from rank %d with segment offset %d.\n", dat->name, rank, obj->size, recv_rank, obj->segment_recv_offset);

#ifdef GPI_VERBOSE  
        printf("Rank %d successfully handled notification from rank %d for exec dat data %s.\n",rank, recv_rank,dat->name);
        fflush(stdout);
#endif
    }
    
#ifdef GPI_VERBOSE
    printf("Rank %d received neccessary exec elements for %s\n",rank,dat->name);
    fflush(stdout);

    printf("Rank %d expects %d non-exec receives from ranks:\n",rank,buff->nonexec_recv_count);
    for(int i=0;i<buff->nonexec_recv_count;i++){
        printf("%d ",buff->nonexec_recv_objs[i].remote_rank);
    }
    printf("\n");
    fflush(stdout);
#endif


    /* TODO move to store elsewhere */
    for(int i=0;i<buff->nonexec_recv_count;i++){
        if(buff->nonexec_recv_objs[i].remote_rank>max_expected_rank)
            max_expected_rank=buff->nonexec_recv_objs[i].remote_rank;
    }


    /* Receive for nonexec elements*/
    for(int i=0;i<buff->nonexec_recv_count;i++){
        GPI_SAFE( gaspi_notify_waitsome(INH_SEGMENT_ID, 
                            dat->index<<7,
                            10, /* Should be max_expected_rank*/
                            &notif_id, /* Notification id should be the dat index*/
                            GPI_TIMEOUT) )


#ifdef GPI_VERBOSE
        printf("Rank %d received non_exec not_ID %d.\n",rank, notif_id);
#endif

        recv_rank = (int) notif_id & ((1<<7)-1); /* Filter out the upper half */
        recv_dat_index= (int) notif_id>>7; /* get only the upper half*/
        
        if(recv_dat_index!=dat->index){
            GPI_FAIL("Accepted nonexec notification from unexpected dat.\n Expected: %d got:%d\n",dat->index, notif_id);
        }
        
        //store and reset notification value
        GPI_SAFE( gaspi_notify_reset(INH_SEGMENT_ID,
                            notif_id,
                            &notif_value) )
        

        //lookup recv object
        int obj_idx=0;
        while(obj_idx < buff->nonexec_recv_count && nonexec_recv_objs[obj_idx].remote_rank != recv_rank)
            obj_idx++;
        
        //check if it didn't find it...
        if(obj_idx >= buff->nonexec_recv_count){
            printf("Dumping exec recv objs:\n");
            printf("Exec recv count: %d\n",buff->nonexec_recv_count);
            for(int i=0;i<buff->nonexec_recv_count;i++){
              op_gpi_recv_obj *obj = &buff->nonexec_recv_objs[i];
              printf(" - remote rank: %d\n - segment_recv_offset: %d\n - memcpy addr: %p\n - size: %d\n\n",obj->remote_rank,obj->segment_recv_offset, obj->memcpy_addr, obj->size);
            }
            fflush(stdout);
            GPI_FAIL("Unable to find nonexec recv object from rank %d in dat %s\n",recv_rank, dat->name); 
        }


        //Use to memcpy data
        op_gpi_recv_obj *obj = &nonexec_recv_objs[obj_idx]; /* not neccessary but looks nicer later*/
        
        // Copy the data into the op_dat->data array
        memcpy(obj->memcpy_addr, (void*) (inh_segment_ptr + obj->segment_recv_offset), obj->size);

        // printf("Rank %d received nonexec data from rank %d for dat %s. XOR: %d, MEMCPYADDR: %p\n", rank, recv_rank, dat->name, xor_byte_summation(obj->memcpy_addr, obj->size), obj->memcpy_addr);

#ifdef GPI_VERBOSE
        printf("Rank %d successfully handled notification from rank %d for nonexec dat data %s.\n",rank,recv_rank,dat->name);
        fflush(stdout);
#endif
    }


#ifdef GPI_VERBOSE
    printf("Rank %d receievd neccessary nonexec elements for %s\n",rank,dat->name);
    fflush(stdout);
#endif

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