rule download_bandpasses:
    output:
        directory("src/data/bandpasses")
    script:
        "src/scripts/download_bandpasses.py"

rule create_observed_catalogs:
    input:
        "src/data/Euclid_trim_27p10_3p5_IR_4NUV.dat"
    output:
        directory("src/data/processed_catalogs")
    script:
        "src/scripts/create_observed_catalogs.py"