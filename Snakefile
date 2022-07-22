rule create_observed_catalogs:
    input:
        "src/data/Euclid_trim_27p10_3p5_IR_4NUV.dat"
    output:
        directory("src/data/observed_catalogs")
    script:
        "src/scripts/create_observed_catalogs.py"