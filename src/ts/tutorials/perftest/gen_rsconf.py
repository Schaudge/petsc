entry="rs_tasks={} rs_cpu={} rs_gpu={} rs_rnum={} nodes={} bindtype={} packdist={} latency={}\n"
bind_opt = ["packed:1", "rs"]
pack_opt = ["packed", "packed"]
latency_opt=["CPU-GPU", "GPU-GPU"]
a_opt=[1,3,6]
c_opt=[1,21,42]
g_opt=[1,3,6]
r_opt=[1,2,6]
n_opt=[1,2,3]

file = open("rsconfigs_test.tab", "w")
#file.write("Configuration 1\n")
file.close()

file = open("rsconfigs_test.tab", "a+")
for i1 in range(len(a_opt)):
    for i2 in range(len(c_opt)):
        for i3 in range(len(g_opt)):
            for i4 in range(len(n_opt)):
                for i5 in range(len(r_opt)):
                    for i6 in range(len(bind_opt)):
                        for i7 in range(len(pack_opt)):
                            for i8 in range(len(latency_opt)):
                            	file.write(entry.format(a_opt[i1],g_opt[i2],c_opt[i3],n_opt[i4],
                                    r_opt[i5],bind_opt[i6],pack_opt[i7],latency_opt[i8]))

file.close()
