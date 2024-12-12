""" Add and manipulate various espresso commands to give a cell relaxation input
"""
from pathlib import Path


def replace(input_lines: list) -> str:
    new_lines = []
    for line in input_lines:

        if 'calculation' in line:
            new_lines.append("   calculation      = 'vc-relax'\n")

        # Expect CONTROL block to be first
        elif 'pseudo_dir' in line:
            new_lines.append(line)
            new_lines.append("   etot_conv_thr = 1e-5\n")
            new_lines.append("   forc_conv_thr = 1e-4\n")

        # Expect empty CELL block
        elif '&CELL' in line:
            new_lines.append(line)
            new_lines.append("   cell_dofree='ibrav'\n")

        else:
            new_lines.append(line)

    return ''.join(new_lines)


if __name__ == "__main__":

    qe_input = 'espresso.pwi'
    source_root = Path("/users/sol/abuccheri/packages/tb_benchmarking/outputs/espresso_e_vs_v")
    target_root = Path("/users/sol/abuccheri/packages/tb_benchmarking/outputs/espresso_relaxation_bulk")

    # silicon done, so removed
    systems = ['bn_cubic', 'bn_hex', 'cdse', 'diamond', 'gan', 'graphite', 'mos2', 'pbs', 'wo3_monoclinic', 'zinc_oxide',
               'gaas', 'germanium', 'mgo', 'nacl', 'tio2_rutile', 'ws2', 'zro2']

    for system in systems:
        with open(Path(source_root, system, '5', qe_input), 'r') as fid:
            input_lines = fid.readlines()

        relax_input: str = replace(input_lines)

        output_dir = Path(target_root, system + '_eq')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / qe_input
        output_file.write_text(relax_input)
