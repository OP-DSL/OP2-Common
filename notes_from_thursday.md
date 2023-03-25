Seems that the problem is that Gaspi_notify_waitsome never works if you try and redo it again. I.e. it only works if the exec/nonexec_recv_count <=1.

Likely because both ranks are sending the same notification ID, so need to encode the rank and the dat-idx into the notification id. 
