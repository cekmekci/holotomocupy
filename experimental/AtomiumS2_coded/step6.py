"""Step 6 (coded aperture) — iterative holotomographic reconstruction, MPI.

Launch with mpirun / mpiexec, one rank per GPU:

    mpirun -np <ngpus> ./bind.sh python step6.py configs/config_bin2.conf

The solver jointly refines the 3-D object, the probe, the coded aperture, and
both position sets (object and code) with the Bilinear-Hessian method.

ONE BIN LEVEL PER INVOCATION, as in the multi-distance pipeline: the
coarse-to-fine ladder lives in run.sh, one config per rung, each reading the
previous rung's checkpoint through `start_iter` / the checkpoint directory.
(The older coded driver ran `--bins 3,2,1` inside a single process instead.)

Checkpoints go to `path_out/checkpoints`; a run resumes automatically from the
latest one if it exists.
"""

import sys

from mpi4py import MPI
import cupy as cp

from holotomocupy.rec_coded_mpi import RecCoded
from holotomocupy.config import parse_args_coded
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader_coded import CodedReader, CodedWriter, find_latest_checkpoint
from holotomocupy.logger_config import logger, set_log_level

cp.cuda.set_pinned_memory_allocator(None)

# --- Parse configuration file -------------------------------------------
args = parse_args_coded(sys.argv[1])
comm = MPI.COMM_WORLD
args.comm = comm
set_log_level(args.log_level)

# --- Distribute object and projection slices across MPI ranks -----------
cl_mpi = MPIClass(comm, args.nzobj, args.ntheta,
                  args.nobj * args.tomo_upsample, 'complex64')

# --- Build I/O helpers --------------------------------------------------
reader = CodedReader(
    args.in_file, args.code_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.nz, args.n, args.ncode, args.bin,
    obj_init_file=args.obj_init_file,
    start_theta=args.start_theta, stride=args.stride,
)
writer = CodedWriter(
    args.path_out, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.nz, args.n, args.ncode,
)

# Physics parameters live in the HDF5 file and are forwarded to the solver.
# z1_code is the exception: the distance that actually focuses the aperture
# differs from the nominal one recorded at the beamline, so it comes from the
# config (see parse_args_coded).
args.energy                  = args.energy if args.energy is not None else reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

# --- Initialise the reconstruction class --------------------------------
logger.info("Create class")
cl = RecCoded(args)
logger.info(f"obj-range  [{cl.st_obj}:{cl.end_obj}), local "
            f"{cl.end_obj - cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_theta}:{cl.end_theta}), local "
            f"{cl.end_theta - cl.st_theta} x {cl.nzobj} x {cl.ndobj}")

# --- Load measurements --------------------------------------------------
# INTENSITY, not its square root: RecCoded.F0 compares K|psi|^2 against d and
# the amplitude model takes its own sqrt.
logger.info("Read data")
reader.read_data(out=cl.data)

# --- Load initial variables ---------------------------------------------
logger.info("Read initial variables")
ckpt = find_latest_checkpoint(args.path_out, args.start_iter)
if ckpt:
    logger.info(f"Resuming from checkpoint: {ckpt}")
    reader.read_checkpoint(ckpt,
                           out_obj=cl.vars['obj'], out_prb=cl.vars['prb'],
                           out_pos=cl.vars['pos'], out_code=cl.vars['code'],
                           out_pos_code=cl.vars['pos_code'])
else:
    reader.read_obj(out=cl.vars['obj'])
    reader.read_prb(out=cl.vars['prb'])
    reader.read_code(out=cl.vars['code'])
    reader.read_pos(out=cl.vars['pos'])
    reader.read_pos_code(out=cl.vars['pos_code'])

# --- Run iterative reconstruction ---------------------------------------
logger.info("Run reconstruction")
vars = cl.BH(writer)
