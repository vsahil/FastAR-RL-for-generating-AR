import os
import numpy as np
import subprocess
from tabulate import tabulate
from typing import List, Callable, Optional


def create_pdf_from_table(filename: str, table_info: np.array, headers: List):
    table = tabulate(table_info, tablefmt="latex", headers=headers)
    # print(table)
    latex = "\\documentclass{article}\n\\pdfpageheight=11in\n\\pdfpagewidth=11in\n\\begin{document}\n" + table + "\\end{document}"
    with open(filename + ".tex", 'w') as f:
        f.write(latex)
    # return
    cmd = ['pdflatex', '-interaction', 'nonstopmode', filename + ".tex"]
    proc = subprocess.Popen(cmd)
    proc.communicate()

    retcode = proc.returncode
    if not retcode == 0:
        os.unlink(filename + ".pdf")
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

    os.unlink(filename + ".tex")
    os.unlink(filename + ".log")
    os.unlink(filename + ".aux")
    os.unlink(filename + ".fdb_latexmk")
    os.unlink(filename + ".fls")
    os.unlink(filename + ".synctex.gz")
    
# headers = ["Benchmarks", "Init Runtime", "IEEE Runtime", "IEEE Speedup", "All double time", "All-double Speed"]
# table = np.array([names, np.around(np.array(init_precs), decimals=3), np.around(np.array(ieee_precs), decimals=3), ieee_speedup, np.around(np.array(all_double_precs), decimals=3), all_double_speedup]).T
