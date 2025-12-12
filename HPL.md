When running the TOP500 CPU benchmark from github, I got the following output on the GB10 with no optimization. 

```bash
ok: [127.0.0.1] => 
    mpirun_output.stdout: |-
        ================================================================================
        HPLinpack 2.3  --  High-Performance Linpack benchmark  --   December 2, 2018
        Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
        Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
        Modified by Julien Langou, University of Colorado Denver
        ================================================================================

        An explanation of the input/output parameters follows:
        T/V    : Wall time / encoded variant.
        N      : The order of the coefficient matrix A.
        NB     : The partitioning blocking factor.
        P      : The number of process rows.
        Q      : The number of process columns.
        Time   : Time in seconds to solve the linear system.
        Gflops : Rate of execution for solving the linear system.

        The following parameter values will be used:

        N      :   98365
        NB     :     256
        PMAP   : Row-major process mapping
        P      :       1
        Q      :      20
        PFACT  :   Right
        NBMIN  :       4
        NDIV   :       2
        RFACT  :   Crout
        BCAST  :  1ringM
        DEPTH  :       1
        SWAP   : Mix (threshold = 64)
        L1     : transposed form
        U      : transposed form
        EQUIL  : yes
        ALIGN  : 8 double precision words

        --------------------------------------------------------------------------------

        - The matrix A is randomly generated for each test.
        - The following scaled residual check will be computed:
              ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
        - The relative machine precision (eps) is taken to be               1.110223e-16
        - Computational tests pass if scaled residuals are less than                16.0

        ================================================================================
        T/V                N    NB     P     Q               Time                 Gflops
        --------------------------------------------------------------------------------
        WR11C2R4       98365   256     1    20            4913.35             1.2914e+02
        HPL_pdgesv() start time Fri Dec 12 10:36:55 2025

        HPL_pdgesv() end time   Fri Dec 12 11:58:48 2025

        --------------------------------------------------------------------------------
        ||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   2.02055227e-03 ...... PASSED
        ================================================================================

        Finished      1 tests with the following results:
                      1 tests completed and passed residual checks,
                      0 tests completed and failed residual checks,
                      0 tests skipped because of illegal input values.
        --------------------------------------------------------------------------------

        End of Tests.
        ================================================================================

PLAY RECAP *****************************************************************************************************
127.0.0.1                  : ok=27   changed=19   unreachable=0    failed=0    skipped=7    rescued=0    ignored=0
