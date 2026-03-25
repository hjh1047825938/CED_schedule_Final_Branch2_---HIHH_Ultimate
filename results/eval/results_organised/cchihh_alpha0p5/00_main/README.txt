Exact CCHIHH files used for conv_shared_bandit_vs_cchihh.
Confirmed from source layout and T500 CV mismatch investigation.

T100 <- ..\\rerun_full\\alpha0.5\\T100
T200 <- ..\\rerun_full\\alpha0.5\\T200
T500 <- ..\\cchihh\\T500\\alpha0.5

Why T100/T200 use rerun_full:
There is no parallel rewritten cchihh\\T100\\alpha0.5 or cchihh\\T200\\alpha0.5 source tree.
Under results/eval/cchihh, only alpha0.5/T50,T300,T400 exist, plus a separately rewritten T500 tree.
So the only canonical alpha0.5 main traces for T100/T200 are rerun_full\\alpha0.5\\T100 and rerun_full\\alpha0.5\\T200.

Why T500 differs:
The T500 CCHIHH files under ..\\cchihh\\T500\\alpha0.5 were rewritten by rewrite_cchihh_t500_all_alpha.py,
so their CV differs from ..\\rerun_full\\alpha0.5\\T500.
