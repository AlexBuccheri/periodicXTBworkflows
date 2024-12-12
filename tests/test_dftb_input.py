from tb_lite.src.dftb_input import Driver, Hamiltonian, Options, generate_dftb_hsd


def test_generate_dftb_hsd():
    driver = Driver(type='ConjugateGradient', lattice_option='No')
    ham = Hamiltonian(method='GFN1-xTB', temperature=0, scc_tolerance=1.e-6, k_grid=[4, 4, 4])
    options = Options()
    input_str = generate_dftb_hsd(driver, ham, options)
    # print(input_str)
    # Fuck string comparisons

