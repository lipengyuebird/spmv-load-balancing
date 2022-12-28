parallelism=4

python ldbc_snb/grb/ic1.py
python ldbc_snb/grb/ic2.py
mpiexec -n ${parallelism} python ldbc_snb/grb_mpi_row/ic1.py
mpiexec -n ${parallelism} python ldbc_snb/grb_mpi_row/ic2.py
mpiexec -n ${parallelism} python ldbc_snb/grb_mpi_nzz/ic1.py
mpiexec -n ${parallelism} python ldbc_snb/grb_mpi_nzz/ic2.py

echo "logs can be found in ldbc_snb/result"